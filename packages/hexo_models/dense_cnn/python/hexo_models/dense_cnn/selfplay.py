"""Game-driven self-play sample generation for dense CNN Model 1.

The self-play loop works from live `hexo_engine.HexoState` objects. It batches
active games through one persistent Rust MCTS session, records a compact
pre-decision sample for every searched position, applies the chosen action
through the engine, writes `.hxr` records, and finalizes targets once each game
reaches a terminal outcome or `max_actions`.

Every active nonterminal position is searched with MCTS over all legal moves.
There are no rollout tails and no progressive widening.

Calibration tunes only the inference/self-play/virtual batch sizes (read from the
trainer); every other search setting comes directly from `config.selfplay`.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from .inference import DenseCNNInference
from .mcts import SearchResult, new_mcts_session
from .performance import _extend_mcts_diagnostic_batches, _summarize_mcts_diagnostic_batches
from .replay import materialize_policy_surprise_rows, write_selfplay_npz
from .samples import Model1SampleData, finalize_game_samples, sample_from_state

import os as _os


def _adaptive_vbatch_enabled() -> bool:
    # Env-gated adaptive virtual_batch_size: hold the per-round leaf budget
    # constant as game concurrency drains (keeps the GPU fed in the tail).
    # Read at runtime so the gate can be toggled per process/epoch. Default off.
    return _os.environ.get("HEXO_ADAPTIVE_VBATCH", "").strip() in ("1", "true", "True")


def generate_selfplay_epoch(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    """Generate one epoch of dense_cnn self-play, writing per-game NPZ shards."""

    trainer = components.model.trainer
    config = trainer.config
    selfplay = config.selfplay
    requested_games = int(games_per_epoch or 0)
    if requested_games < 0:
        raise ValueError("games_per_epoch must be >= 0")

    inference = DenseCNNInference(
        components.model.model,
        device=trainer.device,
        amp=config.training.amp,
        return_logits=False,
        max_batch_size=trainer.inference_batch_size,
        use_trt=config.performance.inference_use_tensorrt,
        bucket_pad_multiple=(config.performance.inference_bucket_pad_multiple or None),
    )
    active_limit = int(trainer.selfplay_batch_size)
    adaptive_vbatch = _adaptive_vbatch_enabled()
    if active_limit <= 0:
        raise ValueError("selfplay active game count must be > 0")
    if active_limit > selfplay.mcts_active_root_limit:
        raise ValueError("selfplay active game count must be <= mcts_active_root_limit")

    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"epoch_{epoch:06d}.hxr"
    horizons = config.architecture.short_term_value_horizons
    base_seed = ctx.config.run.seed or 0

    samples_added = 0
    raw_samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    mcts_search_elapsed = 0.0
    mcts_diagnostic_batches: list[Mapping[str, Any]] = []
    npz_writes: list[Mapping[str, Any]] = []
    started = perf_counter()

    players = (
        HexoRecordPlayer("dense-cnn-a", "player0", "Dense CNN A"),
        HexoRecordPlayer("dense-cnn-b", "player1", "Dense CNN B"),
    )
    mcts_session = new_mcts_session(max_states=selfplay.mcts_session_cache_max_states)
    next_game_index = 0
    active: list[dict[str, Any]] = []
    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        while next_game_index < requested_games or active:
            while len(active) < active_limit and next_game_index < requested_games:
                seed = base_seed + epoch * 1_000_000 + next_game_index
                active.append(
                    {
                        "game_id": f"epoch-{epoch:06d}-selfplay-{next_game_index:06d}",
                        "search_key": next_game_index,
                        "seed": seed,
                        "state": engine.new_game(seed=seed),
                        "pending": [],
                        "actions": [],
                    }
                )
                next_game_index += 1
                games_started += 1

            playable = [
                game
                for game in active
                if engine.terminal(game["state"]) is None and len(game["actions"]) < selfplay.max_actions
            ]
            if playable:
                search_started = perf_counter()
                # Adaptive virtual_batch_size (env-gated): as games finish and
                # concurrency falls, raise leaves-per-root to hold the per-round
                # leaf-request budget (~active_limit * base_vbatch) constant, so
                # forwards stay fat and the GPU stays fed through the drain tail.
                # Bounded by search_visits. Costs a little search quality in the
                # tail (higher vbatch -> more virtual-loss-correlated selection),
                # affecting only the few late, low-concurrency positions.
                effective_vbatch = trainer.mcts_virtual_batch_size
                if adaptive_vbatch and len(playable) > 0:
                    budget = active_limit * trainer.mcts_virtual_batch_size
                    effective_vbatch = max(
                        trainer.mcts_virtual_batch_size,
                        min(int(selfplay.search_visits), -(-budget // len(playable))),
                    )
                searches = mcts_session.run(
                    [game["search_key"] for game in playable],
                    [game["state"] for game in playable],
                    inference,
                    visits=selfplay.search_visits,
                    c_puct=selfplay.c_puct,
                    temperature=selfplay.temperature,
                    seed=base_seed + epoch,
                    virtual_batch_size=effective_vbatch,
                    active_root_limit=selfplay.mcts_active_root_limit,
                    root_dirichlet_total_alpha=(
                        selfplay.root_dirichlet_total_alpha if selfplay.root_dirichlet_noise_enabled else None
                    ),
                    root_dirichlet_noise_fraction=(
                        selfplay.root_dirichlet_noise_fraction if selfplay.root_dirichlet_noise_enabled else None
                    ),
                    root_policy_temperature=selfplay.root_policy_temperature,
                    fpu_reduction=selfplay.fpu_reduction,
                    virtual_loss=selfplay.virtual_loss,
                    widening_policy_mass=selfplay.widening_policy_mass,
                    widening_max_children=selfplay.widening_max_children,
                    widening_min_children=selfplay.widening_min_children,
                )
                mcts_search_elapsed += perf_counter() - search_started
                if len(searches) != len(playable):
                    raise RuntimeError(
                        f"dense_cnn MCTS returned {len(searches)} results for {len(playable)} playable games"
                    )
                _extend_mcts_diagnostic_batches(mcts_diagnostic_batches, searches)
                for game, search in zip(playable, searches):
                    if int(search.visits) != selfplay.search_visits:
                        raise RuntimeError(
                            f"dense_cnn MCTS returned {search.visits} visits; expected exactly {selfplay.search_visits}"
                        )
                    searched_positions += 1
                    mcts_simulations += int(search.visits)
                    state = game["state"]
                    # The sample is captured before the chosen action mutates the
                    # state: policy/legal describe the decision position; outcome
                    # targets are filled once the game ends.
                    sample = sample_from_state(
                        state,
                        game_id=game["game_id"],
                        turn_index=len(game["actions"]),
                        policy=search.visit_policy,
                        root_prior_policy=search.root_prior_policy,
                        metadata={"epoch": epoch, "search_visits": search.visits},
                    )
                    game["pending"].append((sample.current_player, sample, search.root_value))
                    engine.apply_action(state, engine.PlacementAction(unpack_coord_id(search.action_id)))
                    game["actions"].append(search.action_id)

            finished = [
                game
                for game in active
                if engine.terminal(game["state"]) is not None or len(game["actions"]) >= selfplay.max_actions
            ]
            for game in finished:
                terminal = engine.terminal(game["state"])
                truncated = terminal is None
                winner = (
                    _player_label(terminal.winner)
                    if terminal is not None and terminal.winner is not None
                    else None
                )
                writer = record_file.begin_game(game["game_id"], seed=game["seed"])
                for action_id in game["actions"]:
                    writer.record_action(engine.PlacementAction(unpack_coord_id(action_id)))
                if truncated:
                    writer.finish_aborted(
                        AbortRecord(
                            stage="selfplay",
                            exception_type="MaxActionsReached",
                            message=f"dense_cnn self-play reached max_actions={selfplay.max_actions}",
                        )
                    )
                    truncated_games += 1
                else:
                    writer.finish_completed(winner, len(game["actions"]))
                    completed_games += 1

                finalized = _finalize_game_samples(game["pending"], winner, horizons, truncated=truncated)
                materialized, weight_stats = materialize_policy_surprise_rows(
                    finalized,
                    seed=base_seed + epoch * 1_000_000_003 + int(game["search_key"]),
                    uniform_fraction=config.samples.policy_surprise_uniform_fraction,
                    max_weight=config.samples.policy_surprise_max_weight,
                )
                npz_path = record_dir / f"epoch_{epoch:06d}_game_{int(game['search_key']):06d}.npz"
                write_result = write_selfplay_npz(
                    npz_path,
                    materialized,
                    raw_rows=len(finalized),
                    epoch=epoch,
                    game_id=str(game["game_id"]),
                    short_term_value_horizons=horizons,
                )
                raw_samples_added += len(finalized)
                samples_added += len(materialized)
                npz_writes.append(
                    {
                        "path": str(write_result.path),
                        "raw_rows": write_result.raw_rows,
                        "effective_rows": write_result.effective_rows,
                        "policy_surprise_mean": weight_stats["policy_surprise_mean"],
                        "frequency_weight_mean": weight_stats["frequency_weight_mean"],
                    }
                )
                active.remove(game)
                mcts_session.discard(int(game["search_key"]))

    elapsed = perf_counter() - started
    summary = {
        "status": "completed",
        "epoch": epoch,
        "requested_games": requested_games,
        "games_started": games_started,
        "completed_games": completed_games,
        "truncated_games": truncated_games,
        "games_finished": completed_games + truncated_games,
        "raw_samples": raw_samples_added,
        "effective_samples": samples_added,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "search_visits": selfplay.search_visits,
        "selfplay_npz_files": len(npz_writes),
        "record_path": str(record_path),
        "elapsed_seconds": elapsed,
        "mcts_search_elapsed_seconds": mcts_search_elapsed,
        "search_positions_per_second": searched_positions / max(mcts_search_elapsed, 1.0e-9),
        "positions_per_second": searched_positions / max(elapsed, 1.0e-9),
        "active_games": active_limit,
        "mcts_virtual_batch_size": trainer.mcts_virtual_batch_size,
        "mcts_diagnostics": _summarize_mcts_diagnostic_batches(mcts_diagnostic_batches),
        "npz_writes": npz_writes,
    }
    ctx.diagnostics.write_json(f"dense_cnn.selfplay.epoch_{epoch:06d}.json", summary)
    return summary


def _finalize_game_samples(
    pending: list[tuple[str, Model1SampleData, float]],
    winner: str | None,
    horizons: tuple[int, ...],
    *,
    truncated: bool = False,
) -> list[Model1SampleData]:
    return finalize_game_samples(pending, winner, horizons, truncated=truncated)


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
