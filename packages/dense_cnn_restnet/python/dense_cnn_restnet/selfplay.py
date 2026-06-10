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

import queue
import threading
from time import perf_counter, time as wall_clock
from typing import Any, Mapping

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from .inference import DenseCNNInference
from .mcts import SearchResult, _result_from_payload, new_mcts_session
from .performance import _extend_mcts_diagnostic_batches, _summarize_mcts_diagnostic_batches
from .replay import materialize_policy_surprise_rows, write_selfplay_npz
from .samples import SPILL_CATEGORIES, Model1SampleData, count_spill, finalize_game_samples, sample_from_state

import os as _os

# Minimum wall-clock seconds between live-progress writes during a self-play
# epoch. The dashboard polls these to show a live pos/s; the write itself is a
# tiny JSON file so the only reason to throttle is to avoid spamming the disk.
_LIVE_PROGRESS_INTERVAL_SECONDS = 2.0
_LIVE_PROGRESS_NAME = "dense_cnn.selfplay.live.json"


def _move_temperature(
    move_index: int,
    *,
    initial: float,
    final: float,
    decay_moves: int,
    schedule: tuple[tuple[int, float], ...] = (),
    floor: float = 0.0,
    halflife_plies: float = 0.0,
    opening_temperature: float = 0.0,
    opening_moves: int = 0,
) -> float:
    """Temperature for the played move at ply `move_index`.

    With `halflife_plies > 0` (the adaptive scheme, preferred): exponential decay
    `initial * 0.5 ** (move_index / halflife_plies)` clamped at `floor`. The
    caller derives `halflife_plies` from an adaptive expected-game-length EMA, so
    the decay rescales itself as self-play games lengthen/shorten — no absolute
    move anchors to retune. Takes precedence over the anchor schemes below.

    Otherwise, if `schedule` (a tuple of ``(move, temperature)`` anchors) is
    given: piecewise-linear interpolation between anchors, the first anchor's
    value before it, and the final segment's slope continued past the last anchor
    down to `floor` (then held). Otherwise: linear decay from `initial` (ply 0) to
    `final` (ply >= `decay_moves`), held flat afterwards (`decay_moves <= 0` keeps
    `initial`). Either way the opening explores and the endgame sharpens.

    Opening anchor: when `opening_temperature > 0` and `move_index < opening_moves`,
    the result is floored at `opening_temperature` (i.e. `max(opening_temperature,
    base)`). This holds a higher/flatter temperature over the first few decisions to
    diversify the opening, BEFORE the adaptive decay takes over — it never sharpens
    the opening below the base curve. Applies on top of whichever base scheme is
    active above.
    """

    if halflife_plies > 0.0:
        base = max(floor, initial * 0.5 ** (move_index / halflife_plies))
    elif schedule:
        base = _scheduled_temperature(move_index, schedule, floor)
    elif decay_moves <= 0:
        base = initial
    else:
        fraction = min(move_index, decay_moves) / decay_moves
        base = initial + (final - initial) * fraction

    if opening_temperature > 0.0 and move_index < opening_moves:
        return max(opening_temperature, base)
    return base


def _scheduled_temperature(
    move_index: int, schedule: tuple[tuple[int, float], ...], floor: float
) -> float:
    """Piecewise-linear temperature from ``(move, temperature)`` anchors.

    Held flat at the first anchor before it; linearly interpolated between
    anchors; past the last anchor the final segment's slope continues down to
    `floor`, then holds. Anchors are assumed sorted with unique ascending moves
    (enforced by ``config._parse_temperature_schedule``).
    """

    first_move, first_temp = schedule[0]
    if move_index <= first_move:
        return first_temp
    for index in range(1, len(schedule)):
        m0, t0 = schedule[index - 1]
        m1, t1 = schedule[index]
        if move_index <= m1:
            return t0 + (t1 - t0) * (move_index - m0) / (m1 - m0)
    if len(schedule) >= 2:
        m0, t0 = schedule[-2]
        m1, t1 = schedule[-1]
        slope = (t1 - t0) / (m1 - m0) if m1 != m0 else 0.0
        return max(floor, t1 + slope * (move_index - m1))
    return max(floor, schedule[-1][1])


# Adaptive expected-game-length state for the half-life temperature scheme: an
# EMA of the measured mean decisions/game, persisted per run so it survives
# process bounces and resumes alongside the run itself (not the checkpoint —
# it is a property of current self-play, not of the weights).
_LENGTH_EMA_NAME = "length_ema.json"
_LENGTH_EMA_DECAY = 0.75  # ema = 0.75 * ema + 0.25 * latest epoch's mean length


def _read_length_ema(record_dir: Any, prior: float) -> float:
    try:
        import json

        payload = json.loads((record_dir / _LENGTH_EMA_NAME).read_text(encoding="utf-8"))
        value = float(payload["mean_game_length_ema"])
        return value if value > 0.0 else float(prior)
    except (OSError, KeyError, TypeError, ValueError):
        return float(prior)


def _write_length_ema(record_dir: Any, value: float, *, epoch: int, latest_mean: float) -> None:
    import json

    (record_dir / _LENGTH_EMA_NAME).write_text(
        json.dumps(
            {
                "mean_game_length_ema": float(value),
                "latest_epoch": int(epoch),
                "latest_mean_game_length": float(latest_mean),
                "decay": _LENGTH_EMA_DECAY,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _adaptive_vbatch_enabled() -> bool:
    # Env-gated adaptive virtual_batch_size: hold the per-round leaf budget
    # constant as game concurrency drains (keeps the GPU fed in the tail).
    # Read at runtime so the gate can be toggled per process/epoch. Default off.
    return _os.environ.get("HEXO_ADAPTIVE_VBATCH", "").strip() in ("1", "true", "True")


def generate_selfplay_epoch(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    trainer = components.model.trainer
    scheduler = getattr(trainer.config.selfplay, "scheduler", "lockstep")
    if scheduler == "continuous":
        return _generate_selfplay_epoch_continuous(
            ctx=ctx,
            components=components,
            epoch=epoch,
            games_per_epoch=games_per_epoch,
        )
    if scheduler != "lockstep":
        raise ValueError(f"unsupported selfplay scheduler {scheduler!r}")
    return _generate_selfplay_epoch_lockstep(
        ctx=ctx,
        components=components,
        epoch=epoch,
        games_per_epoch=games_per_epoch,
    )


def _generate_selfplay_epoch_lockstep(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
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
        trt_allow_fallback=config.performance.inference_trt_allow_torch_fallback,
        fp16_model=config.performance.inference_fp16_model,
        fp16_allow_fallback=config.performance.inference_fp16_allow_fallback,
        use_torch_compile=config.performance.inference_use_torch_compile,
        compile_allow_fallback=config.performance.inference_compile_allow_torch_fallback,
        attention_kv_gather=config.performance.attention_kv_gather,
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

    # Adaptive temperature half-life: a fraction of the expected game length
    # (EMA of measured mean decisions/game, seeded with the config prior on a
    # fresh run). 0 disables the scheme and falls back to the anchor schedule.
    expected_game_length = _read_length_ema(record_dir, selfplay.temperature_length_prior)
    temperature_halflife_plies = (
        selfplay.temperature_halflife_fraction * expected_game_length
        if selfplay.temperature_halflife_fraction > 0.0
        else 0.0
    )

    samples_added = 0
    raw_samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    mcts_search_elapsed = 0.0
    # Observation-only spill telemetry: per-category facts beyond hex distance 20
    # from the crop center, which the fixed radius-20 disk crop cannot represent
    # (Spec A). Accumulated over every searched position's sample.
    epoch_spill = {category: 0 for category in SPILL_CATEGORIES}
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
    # Per-round search seed counter. The native session derives its root-Dirichlet
    # noise RNG and the played-move sampling uniform from (seed, batch index) ONLY,
    # so a seed held constant across the epoch (the old `base_seed + epoch`) reused
    # the SAME noise realization and the SAME selection quantile at every move of a
    # game (and across games sharing a batch slot). Each game makes at most one
    # move per `run` call, so bumping the seed per round restores the per-move
    # i.i.d. draws the AlphaZero/KataGo exploration design assumes, while staying
    # fully deterministic in (run seed, epoch, round).
    search_rounds = 0

    last_live_write = 0.0

    def _write_live_progress(status: str) -> None:
        # Snapshot of the in-progress epoch so the dashboard can show a live
        # pos/s. The authoritative throughput figure is the same one the
        # completed-epoch summary and calibration report
        # (searched_positions / mcts_search_elapsed), so the live and final
        # numbers are consistent. Overwrites a single per-run file; a wall-clock
        # timestamp lets the reader detect a stale (run-ended) file.
        now = perf_counter()
        ctx.diagnostics.write_json(
            _LIVE_PROGRESS_NAME,
            {
                "status": status,
                "epoch": epoch,
                "timestamp": wall_clock(),
                "requested_games": requested_games,
                "games_started": games_started,
                "completed_games": completed_games,
                "truncated_games": truncated_games,
                "games_finished": completed_games + truncated_games,
                "active_games": len(active),
                "active_limit": active_limit,
                "searched_positions": searched_positions,
                "mcts_simulations": mcts_simulations,
                "raw_samples": raw_samples_added,
                "effective_samples": samples_added,
                "elapsed_seconds": now - started,
                "mcts_search_elapsed_seconds": mcts_search_elapsed,
                "search_positions_per_second": searched_positions / max(mcts_search_elapsed, 1.0e-9),
                "positions_per_second": searched_positions / max(now - started, 1.0e-9),
            },
        )

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
                # Per-move temperature decay: each playable game is at its own ply
                # (len(actions)), so the played-move temperature is resolved per
                # game and passed as a vector aligned with the playable order.
                move_temperatures = [
                    _move_temperature(
                        len(game["actions"]),
                        initial=selfplay.temperature,
                        final=selfplay.final_temperature,
                        decay_moves=selfplay.temperature_decay_moves,
                        schedule=selfplay.temperature_schedule,
                        floor=selfplay.temperature_floor,
                        halflife_plies=temperature_halflife_plies,
                        opening_temperature=selfplay.opening_temperature,
                        opening_moves=selfplay.opening_moves,
                    )
                    for game in playable
                ]
                searches = mcts_session.run(
                    [game["search_key"] for game in playable],
                    [game["state"] for game in playable],
                    inference,
                    visits=selfplay.search_visits,
                    c_puct=selfplay.c_puct,
                    temperature=selfplay.temperature,
                    seed=base_seed + epoch * 1_000_003 + search_rounds,
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
                    forced_playout_k=selfplay.forced_playout_k,
                    move_temperatures=move_temperatures,
                )
                mcts_search_elapsed += perf_counter() - search_started
                search_rounds += 1
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
                    for category, count in count_spill(sample).items():
                        epoch_spill[category] += count
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

            if perf_counter() - last_live_write >= _LIVE_PROGRESS_INTERVAL_SECONDS:
                _write_live_progress("running")
                last_live_write = perf_counter()

    elapsed = perf_counter() - started
    games_done = completed_games + truncated_games
    if games_done > 0:
        latest_mean_length = raw_samples_added / games_done
        updated_ema = (
            _LENGTH_EMA_DECAY * expected_game_length + (1.0 - _LENGTH_EMA_DECAY) * latest_mean_length
        )
        _write_length_ema(record_dir, updated_ema, epoch=epoch, latest_mean=latest_mean_length)
    summary = {
        "status": "completed",
        "epoch": epoch,
        "temperature_control": {
            "expected_game_length": expected_game_length,
            "halflife_plies": temperature_halflife_plies,
            "halflife_fraction": selfplay.temperature_halflife_fraction,
        },
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
        # Per-category facts the radius-20 hex-disk crop cannot represent (spill),
        # plus the total. Observation only (Spec A); never affects training.
        "spill": {**epoch_spill, "total": sum(epoch_spill.values())},
    }
    ctx.diagnostics.write_json(f"dense_cnn.selfplay.epoch_{epoch:06d}.json", summary)
    # Final live snapshot so the dashboard reflects the just-finished epoch's
    # numbers (status "completed") until the next epoch's self-play begins.
    _write_live_progress("completed")
    return summary


def _generate_selfplay_epoch_continuous(*, ctx: Any, components: Any, epoch: int, games_per_epoch: int) -> dict[str, Any]:
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
        trt_allow_fallback=config.performance.inference_trt_allow_torch_fallback,
        fp16_model=config.performance.inference_fp16_model,
        fp16_allow_fallback=config.performance.inference_fp16_allow_fallback,
        use_torch_compile=config.performance.inference_use_torch_compile,
        compile_allow_fallback=config.performance.inference_compile_allow_torch_fallback,
        attention_kv_gather=config.performance.attention_kv_gather,
    )
    active_limit = int(trainer.selfplay_batch_size)
    if active_limit <= 0:
        raise ValueError("selfplay active game count must be > 0")
    if active_limit > selfplay.mcts_active_root_limit:
        raise ValueError("selfplay active game count must be <= mcts_active_root_limit")

    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"epoch_{epoch:06d}.hxr"
    horizons = config.architecture.short_term_value_horizons
    base_seed = ctx.config.run.seed or 0
    expected_game_length = _read_length_ema(record_dir, selfplay.temperature_length_prior)
    temperature_halflife_plies = (
        selfplay.temperature_halflife_fraction * expected_game_length
        if selfplay.temperature_halflife_fraction > 0.0
        else 0.0
    )
    temperature_by_ply = [
        _move_temperature(
            ply,
            initial=selfplay.temperature,
            final=selfplay.final_temperature,
            decay_moves=selfplay.temperature_decay_moves,
            schedule=selfplay.temperature_schedule,
            floor=selfplay.temperature_floor,
            halflife_plies=temperature_halflife_plies,
        )
        for ply in range(max(1, int(selfplay.max_actions) + 1))
    ]

    samples_added = 0
    raw_samples_added = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    epoch_spill = {category: 0 for category in SPILL_CATEGORIES}
    npz_writes: list[Mapping[str, Any]] = []
    started = perf_counter()
    mcts_search_elapsed = 0.0
    last_live_write = 0.0
    next_game_index = 0
    active: dict[int, dict[str, Any]] = {}
    writer_errors: list[BaseException] = []
    writer_failed = threading.Event()
    writer_results: list[Mapping[str, Any]] = []
    write_queue: queue.Queue[Any] = queue.Queue()

    players = (
        HexoRecordPlayer("dense-cnn-a", "player0", "Dense CNN A"),
        HexoRecordPlayer("dense-cnn-b", "player1", "Dense CNN B"),
    )

    def _write_live_progress(status: str) -> None:
        now = perf_counter()
        ctx.diagnostics.write_json(
            _LIVE_PROGRESS_NAME,
            {
                "status": status,
                "epoch": epoch,
                "timestamp": wall_clock(),
                "requested_games": requested_games,
                "games_started": games_started,
                "completed_games": completed_games,
                "truncated_games": truncated_games,
                "games_finished": completed_games + truncated_games,
                "active_games": len(active),
                "active_limit": active_limit,
                "searched_positions": searched_positions,
                "mcts_simulations": mcts_simulations,
                "raw_samples": raw_samples_added,
                "effective_samples": samples_added,
                "elapsed_seconds": now - started,
                # The whole continuous epoch runs inside one run_continuous call,
                # so wall time IS search time; mcts_search_elapsed is only filled
                # in after the call returns (mid-epoch it would read ~0 and blow
                # up the live rate, which is exactly what the dashboard polls).
                "mcts_search_elapsed_seconds": now - started,
                "search_positions_per_second": searched_positions / max(now - started, 1.0e-9),
                "positions_per_second": searched_positions / max(now - started, 1.0e-9),
                "scheduler": "continuous",
            },
        )

    def _new_game(game_index: int) -> dict[str, Any]:
        seed = base_seed + epoch * 1_000_000 + game_index
        return {
            "game_id": f"epoch-{epoch:06d}-selfplay-{game_index:06d}",
            "search_key": game_index,
            "seed": seed,
            "state": engine.new_game(seed=seed),
            "pending": [],
            "actions": [],
        }

    def _writer(record_file: Any) -> None:
        while True:
            item = write_queue.get()
            try:
                if item is None:
                    return
                if writer_failed.is_set():
                    continue
                try:
                    game = item["game"]
                    winner = item["winner"]
                    truncated = bool(item["truncated"])
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
                    else:
                        writer.finish_completed(winner, len(game["actions"]))
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
                    writer_results.append(
                        {
                            "path": str(write_result.path),
                            "raw_rows": write_result.raw_rows,
                            "effective_rows": write_result.effective_rows,
                            "policy_surprise_mean": weight_stats["policy_surprise_mean"],
                            "frequency_weight_mean": weight_stats["frequency_weight_mean"],
                        }
                    )
                except BaseException as exc:
                    if not writer_errors:
                        writer_errors.append(exc)
                    writer_failed.set()
            finally:
                write_queue.task_done()

    while len(active) < active_limit and next_game_index < requested_games:
        game = _new_game(next_game_index)
        active[int(game["search_key"])] = game
        next_game_index += 1
        games_started += 1

    scheduler_summary: Mapping[str, Any] = {
        "flush_count": 0,
        "queued_states": 0,
        "flushed_states": 0,
        "mean_flush_states": 0.0,
        "no_progress_flushes": 0,
        "moves_decided": 0,
        "flush_size_histogram": {},
        "on_move_seconds": 0.0,
    }

    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        writer_thread = threading.Thread(target=_writer, args=(record_file,), name="dense-cnn-continuous-writer")
        writer_thread.start()

        def _on_move(game_key: int, payload: Mapping[str, Any]) -> object:
            nonlocal searched_positions, mcts_simulations, completed_games, truncated_games
            nonlocal next_game_index, games_started, last_live_write
            game = active.pop(int(game_key), None)
            if game is None:
                raise RuntimeError(f"continuous MCTS callback received unknown game key {game_key}")
            if perf_counter() - last_live_write >= _LIVE_PROGRESS_INTERVAL_SECONDS:
                _write_live_progress("running")
                last_live_write = perf_counter()
            search = _result_from_payload(payload)
            if int(search.visits) != selfplay.search_visits:
                raise RuntimeError(
                    f"dense_cnn MCTS returned {search.visits} visits; expected exactly {selfplay.search_visits}"
                )
            searched_positions += 1
            mcts_simulations += int(search.visits)
            state = game["state"]
            sample = sample_from_state(
                state,
                game_id=game["game_id"],
                turn_index=len(game["actions"]),
                policy=search.visit_policy,
                root_prior_policy=search.root_prior_policy,
                metadata={"epoch": epoch, "search_visits": search.visits},
            )
            for category, count in count_spill(sample).items():
                epoch_spill[category] += count
            game["pending"].append((sample.current_player, sample, search.root_value))
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(search.action_id)))
            game["actions"].append(search.action_id)

            terminal = engine.terminal(state)
            truncated = terminal is None and len(game["actions"]) >= selfplay.max_actions
            if terminal is None and not truncated:
                active[int(game_key)] = game
                return ("advance", state)

            winner = (
                _player_label(terminal.winner)
                if terminal is not None and terminal.winner is not None
                else None
            )
            if truncated:
                truncated_games += 1
            else:
                completed_games += 1
            write_queue.put({"game": game, "winner": winner, "truncated": truncated})

            if next_game_index < requested_games:
                replacement = _new_game(next_game_index)
                next_game_index += 1
                games_started += 1
                active[int(replacement["search_key"])] = replacement
                return ("replace", int(replacement["search_key"]), replacement["state"])
            if perf_counter() - last_live_write >= _LIVE_PROGRESS_INTERVAL_SECONDS:
                _write_live_progress("running")
                last_live_write = perf_counter()
            return None

        mcts_session = new_mcts_session(max_states=selfplay.mcts_session_cache_max_states)
        try:
            if active:
                search_started = perf_counter()
                # Calibration normally selects the virtual batch; mirror the
                # lockstep default (visits) when it is absent so both schedulers
                # degrade identically without calibration.
                virtual_batch_size = int(trainer.mcts_virtual_batch_size or selfplay.search_visits)
                flush_target = int(selfplay.scheduler_flush_target or trainer.inference_batch_size)
                flush_target = max(1, min(flush_target, len(active) * virtual_batch_size))
                scheduler_summary = mcts_session.run_continuous(
                    [int(key) for key in active],
                    [game["state"] for game in active.values()],
                    inference,
                    _on_move,
                    visits=selfplay.search_visits,
                    c_puct=selfplay.c_puct,
                    base_seed=base_seed + epoch * 1_000_003,
                    virtual_batch_size=virtual_batch_size,
                    flush_target=flush_target,
                    active_root_limit=selfplay.mcts_active_root_limit,
                    temperature_by_ply=temperature_by_ply,
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
                    forced_playout_k=selfplay.forced_playout_k,
                )
                mcts_search_elapsed += perf_counter() - search_started
        finally:
            write_queue.put(None)
            write_queue.join()
            writer_thread.join()
        if writer_errors:
            raise RuntimeError("continuous self-play writer failed") from writer_errors[0]
        if active:
            # run_continuous returns only when every slot is Empty; games left in
            # `active` mean the scheduler dropped work — never accept a silently
            # short epoch.
            raise RuntimeError(
                f"continuous self-play ended with {len(active)} unfinished games "
                f"({games_started} started, {completed_games + truncated_games} finished)"
            )

    npz_writes.extend(writer_results)
    raw_samples_added = sum(int(item["raw_rows"]) for item in npz_writes)
    samples_added = sum(int(item["effective_rows"]) for item in npz_writes)
    elapsed = perf_counter() - started
    games_done = completed_games + truncated_games
    if games_done > 0:
        latest_mean_length = raw_samples_added / games_done
        updated_ema = (
            _LENGTH_EMA_DECAY * expected_game_length + (1.0 - _LENGTH_EMA_DECAY) * latest_mean_length
        )
        _write_length_ema(record_dir, updated_ema, epoch=epoch, latest_mean=latest_mean_length)
    # The Rust scheduler returns one epoch-wide diagnostics aggregate (the same
    # {tree, evaluation} shape as a lockstep batch); per-move payloads carry only
    # root diagnostics. Split the aggregate out of the scheduler section so the
    # epoch JSON does not store the same blob twice.
    scheduler_summary = dict(scheduler_summary)
    batch_diagnostics = scheduler_summary.pop("mcts_batch_diagnostics", None) or {}
    summary = {
        "status": "completed",
        "epoch": epoch,
        "scheduler": "continuous",
        "scheduler_diagnostics": scheduler_summary,
        "temperature_control": {
            "expected_game_length": expected_game_length,
            "halflife_plies": temperature_halflife_plies,
            "halflife_fraction": selfplay.temperature_halflife_fraction,
        },
        "requested_games": requested_games,
        "games_started": games_started,
        "completed_games": completed_games,
        "truncated_games": truncated_games,
        "games_finished": games_done,
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
        "mcts_diagnostics": _summarize_mcts_diagnostic_batches([batch_diagnostics]),
        "npz_writes": npz_writes,
        "spill": {**epoch_spill, "total": sum(epoch_spill.values())},
    }
    ctx.diagnostics.write_json(f"dense_cnn.selfplay.epoch_{epoch:06d}.json", summary)
    _write_live_progress("completed")
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
