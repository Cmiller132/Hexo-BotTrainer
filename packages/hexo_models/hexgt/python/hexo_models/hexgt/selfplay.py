"""Game-driven self-play sample generation for hexgt (Model 2).

Mirrors `dense_cnn/selfplay.py`'s game-driven loop (no sample budget: request
`games_per_epoch` complete games, keep `active_games` in flight, search every
playable nonterminal position to terminal / `max_actions`), but drives the
dynamic-GNN `HexgtMctsSession` + `HexgtInference` evaluator and writes the SAME
representation-agnostic COMPACT shard format `expand.py` already reads
(`dense_cnn.compact_io.write_compact_shard`). The training read path
(`HexgtTrainer.train_on_shards` -> `expand.build_training_batch`) consumes those
shards directly, so a self-play epoch closes the RL loop without any new schema.

LEAF BATCHING (the Phase-7 "async leaf batcher" intent, pragmatically). The
binding Phase-7 decision targeted an asynchronous Rust leaf-eval batcher, but its
own sequencing rule ("get a baseline WORKING first, don't over-build") plus the
Phase-5d result (the synchronous session already hits 29.7 pos/s, beating
dense_cnn 96x8's ~23) make the proven path the right one here: the existing
`HexgtMctsSession.run` already coalesces the in-flight leaves of *all* concurrent
games into one size-bucketed GPU forward per round (via `virtual_batch_size` +
`active_root_limit`) — i.e. many concurrent games feeding one shared batched eval,
which is the throughput property the async batcher was meant to deliver. We run
self-play on it and revisit a true work-stealing async batcher only if a converged
run proves throughput-starved (none of the gates suggest it will). Determinism of
batch composition is irrelevant for RL self-play (noise + temperature dominate).

Candidate radius `n` is threaded into the session so training support == search
expansion. Sample facts / value / opp-policy / short-term-value targets reuse
dense_cnn's already-finalized `Model1SampleData` machinery (per per-placement
turn), exactly as `expand.py` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

from hexo_models.dense_cnn.compact_io import write_compact_shard
from hexo_models.dense_cnn.samples import (
    Model1SampleData,
    finalize_game_samples,
    sample_from_state,
)

from .config import HexgtConfig
from .inference import HexgtInference
from .mcts import HexgtMctsSession, new_mcts_session


def _move_temperature(
    move_index: int,
    *,
    initial: float,
    final: float,
    decay_moves: int,
    schedule: tuple[tuple[int, float], ...] = (),
    floor: float = 0.0,
) -> float:
    """Played-move temperature at ply `move_index` (copy of dense_cnn's rule).

    `schedule` (sorted ``(move, temp)`` anchors) takes precedence: piecewise
    linear, held at the first anchor before it, the final slope continued past
    the last anchor down to `floor`. Otherwise linear decay `initial`->`final`
    over `decay_moves`, held flat after. The opening explores, the endgame sharpens.
    """

    if schedule:
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
    if decay_moves <= 0:
        return initial
    fraction = min(move_index, decay_moves) / decay_moves
    return initial + (final - initial) * fraction


def _policy_entropy(pairs: Sequence[tuple[int, float]]) -> tuple[float, float, int]:
    """(Shannon entropy in nats, top-1 mass fraction, support size) of a visit
    or prior distribution. Robust to unnormalized weights (renormalizes)."""

    weights = [float(w) for _a, w in pairs if float(w) > 0.0]
    total = sum(weights)
    if total <= 0.0 or not weights:
        return 0.0, 0.0, 0
    ent = 0.0
    top = 0.0
    for w in weights:
        p = w / total
        ent -= p * log(p)
        if p > top:
            top = p
    return ent, top, len(weights)


@dataclass
class SelfPlayResult:
    """Summary of one self-play batch + the shards it wrote + example traces."""

    epoch: int
    requested_games: int
    completed_games: int
    truncated_games: int
    raw_samples: int
    searched_positions: int
    mcts_simulations: int
    elapsed_seconds: float
    mcts_search_elapsed_seconds: float
    positions_per_second: float
    search_positions_per_second: float
    shard_paths: list[str] = field(default_factory=list)
    example_games: list[dict[str, Any]] = field(default_factory=list)
    # Aggregate play-style signal across all searched positions.
    mean_visit_entropy: float = 0.0
    mean_prior_entropy: float = 0.0
    mean_top_visit_fraction: float = 0.0
    mean_candidate_count: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "requested_games": self.requested_games,
            "completed_games": self.completed_games,
            "truncated_games": self.truncated_games,
            "raw_samples": self.raw_samples,
            "searched_positions": self.searched_positions,
            "mcts_simulations": self.mcts_simulations,
            "elapsed_seconds": self.elapsed_seconds,
            "mcts_search_elapsed_seconds": self.mcts_search_elapsed_seconds,
            "positions_per_second": self.positions_per_second,
            "search_positions_per_second": self.search_positions_per_second,
            "selfplay_npz_files": len(self.shard_paths),
            "mean_visit_entropy": self.mean_visit_entropy,
            "mean_prior_entropy": self.mean_prior_entropy,
            "mean_top_visit_fraction": self.mean_top_visit_fraction,
            "mean_candidate_count": self.mean_candidate_count,
        }


def run_selfplay_games(
    model: Any,
    config: HexgtConfig,
    *,
    num_games: int,
    output_dir: Path,
    epoch: int,
    device: str = "cuda",
    fp16: bool = True,
    base_seed: int = 0,
    active_games: int | None = None,
    virtual_batch_size: int | None = None,
    collect_examples: int = 2,
    progress: Callable[[str], None] | None = None,
) -> SelfPlayResult:
    """Play `num_games` self-play games, writing one compact shard per game.

    `model` is already on `device` (and may be ``torch.compile``d). The compact
    shards land in `output_dir` named ``epoch_{epoch:06d}_game_{i:06d}.npz`` —
    the dense_cnn compact layout `expand.py` reads. Returns a `SelfPlayResult`
    with throughput + play-style aggregates + a few full example-game traces.
    """

    if num_games < 0:
        raise ValueError("num_games must be >= 0")
    selfplay = config.selfplay
    horizons = tuple(int(h) for h in config.architecture.short_term_value_horizons)
    n = int(config.architecture.candidate_radius)
    active_limit = int(active_games or selfplay.active_games)
    active_limit = max(1, min(active_limit, selfplay.mcts_active_root_limit))
    vbatch = virtual_batch_size if virtual_batch_size is not None else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inference = HexgtInference(model, device=device, fp16=fp16)
    mcts_session = new_mcts_session(
        max_states=selfplay.mcts_session_cache_max_states, n=n
    )

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    completed_games = 0
    truncated_games = 0
    raw_samples = 0
    searched_positions = 0
    mcts_simulations = 0
    mcts_search_elapsed = 0.0
    shard_paths: list[str] = []
    example_games: list[dict[str, Any]] = []
    ent_visit_sum = 0.0
    ent_prior_sum = 0.0
    top_visit_sum = 0.0
    cand_count_sum = 0
    next_game_index = 0
    active: list[dict[str, Any]] = []
    started = perf_counter()

    while next_game_index < num_games or active:
        while len(active) < active_limit and next_game_index < num_games:
            seed = base_seed + epoch * 1_000_000 + next_game_index
            active.append(
                {
                    "game_id": f"epoch-{epoch:06d}-selfplay-{next_game_index:06d}",
                    "search_key": next_game_index,
                    "seed": seed,
                    "state": engine.new_game(seed=seed),
                    "pending": [],
                    "actions": [],
                    "trace": [] if next_game_index < collect_examples else None,
                }
            )
            next_game_index += 1

        playable = [
            game
            for game in active
            if engine.terminal(game["state"]) is None
            and len(game["actions"]) < selfplay.max_actions
        ]
        if playable:
            move_temperatures = [
                _move_temperature(
                    len(game["actions"]),
                    initial=selfplay.temperature,
                    final=selfplay.final_temperature,
                    decay_moves=selfplay.temperature_decay_moves,
                    schedule=selfplay.temperature_schedule,
                    floor=selfplay.temperature_floor,
                )
                for game in playable
            ]
            search_started = perf_counter()
            searches = mcts_session.run(
                [game["search_key"] for game in playable],
                [game["state"] for game in playable],
                inference,
                visits=selfplay.search_visits,
                c_puct=selfplay.c_puct,
                temperature=selfplay.temperature,
                seed=base_seed + epoch,
                virtual_batch_size=vbatch,
                active_root_limit=selfplay.mcts_active_root_limit,
                root_dirichlet_total_alpha=(
                    selfplay.root_dirichlet_total_alpha
                    if selfplay.root_dirichlet_noise_enabled
                    else None
                ),
                root_dirichlet_noise_fraction=(
                    selfplay.root_dirichlet_noise_fraction
                    if selfplay.root_dirichlet_noise_enabled
                    else None
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
            if len(searches) != len(playable):
                raise RuntimeError(
                    f"hexgt MCTS returned {len(searches)} results for {len(playable)} playable games"
                )
            for game, search, temp in zip(playable, searches, move_temperatures):
                searched_positions += 1
                mcts_simulations += int(search.visits)
                state = game["state"]
                visit_pairs = list(search.visit_policy)
                prior_pairs = list(search.root_prior_policy)
                v_ent, v_top, v_support = _policy_entropy(visit_pairs)
                p_ent, _p_top, _p_support = _policy_entropy(prior_pairs)
                ent_visit_sum += v_ent
                ent_prior_sum += p_ent
                top_visit_sum += v_top
                cand_count_sum += len(prior_pairs)
                sample = sample_from_state(
                    state,
                    game_id=game["game_id"],
                    turn_index=len(game["actions"]),
                    policy=visit_pairs,
                    root_prior_policy=prior_pairs,
                    metadata={"epoch": epoch, "search_visits": search.visits},
                )
                game["pending"].append((sample.current_player, sample, search.root_value))
                if game["trace"] is not None:
                    game["trace"].append(
                        {
                            "move": len(game["actions"]),
                            "player": str(sample.current_player),
                            "action_id": int(search.action_id),
                            "root_value": float(search.root_value),
                            "visits": int(search.visits),
                            "candidates": int(len(prior_pairs)),
                            "visit_entropy": float(v_ent),
                            "prior_entropy": float(p_ent),
                            "top_visit_fraction": float(v_top),
                            "visit_support": int(v_support),
                            "temperature": float(temp),
                        }
                    )
                engine.apply_action(
                    state, engine.PlacementAction(unpack_coord_id(search.action_id))
                )
                game["actions"].append(search.action_id)

        finished = [
            game
            for game in active
            if engine.terminal(game["state"]) is not None
            or len(game["actions"]) >= selfplay.max_actions
        ]
        for game in finished:
            terminal = engine.terminal(game["state"])
            truncated = terminal is None
            winner = (
                _player_label(terminal.winner)
                if terminal is not None and terminal.winner is not None
                else None
            )
            finalized = finalize_game_samples(
                game["pending"], winner, horizons, truncated=truncated
            )
            npz_path = output_dir / f"epoch_{epoch:06d}_game_{int(game['search_key']):06d}.npz"
            write_compact_shard(npz_path, finalized, short_term_value_horizons=horizons)
            shard_paths.append(str(npz_path))
            raw_samples += len(finalized)
            if truncated:
                truncated_games += 1
            else:
                completed_games += 1
            if game["trace"] is not None:
                example_games.append(
                    {
                        "game_id": game["game_id"],
                        "winner": winner,
                        "truncated": truncated,
                        "turns": len(game["actions"]),
                        "moves": game["trace"],
                    }
                )
            active.remove(game)
            mcts_session.discard(int(game["search_key"]))

        if playable:
            _log(
                f"  selfplay: {completed_games + truncated_games}/{num_games} games, "
                f"{searched_positions} pos, "
                f"{searched_positions / max(perf_counter() - started, 1e-9):.1f} pos/s"
            )

    elapsed = perf_counter() - started
    sp = max(1, searched_positions)
    return SelfPlayResult(
        epoch=epoch,
        requested_games=num_games,
        completed_games=completed_games,
        truncated_games=truncated_games,
        raw_samples=raw_samples,
        searched_positions=searched_positions,
        mcts_simulations=mcts_simulations,
        elapsed_seconds=elapsed,
        mcts_search_elapsed_seconds=mcts_search_elapsed,
        positions_per_second=searched_positions / max(elapsed, 1e-9),
        search_positions_per_second=searched_positions / max(mcts_search_elapsed, 1e-9),
        shard_paths=shard_paths,
        example_games=example_games,
        mean_visit_entropy=ent_visit_sum / sp,
        mean_prior_entropy=ent_prior_sum / sp,
        mean_top_visit_fraction=top_visit_sum / sp,
        mean_candidate_count=cand_count_sum / sp,
    )


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))


def generate_selfplay_epoch(
    *, ctx: Any, components: Any, epoch: int, games_per_epoch: int
) -> dict[str, Any]:
    """Plugin hook: one epoch of hexgt self-play, writing per-game compact shards.

    Mirrors `dense_cnn`'s `generate_selfplay` signature so the generic training
    pipeline can drive it. Writes compact shards under
    ``ctx.output_dir/selfplay`` and a JSON summary into diagnostics.
    """

    trainer = components.model.trainer
    config: HexgtConfig = trainer.config
    model = components.model.model
    device = str(next(model.parameters()).device)
    base_seed = int(getattr(getattr(ctx, "config", None), "run", None).seed or 0) if getattr(ctx, "config", None) else 0
    vbatch = getattr(trainer, "mcts_virtual_batch_size", None)
    active = int(getattr(trainer, "selfplay_batch_size", 0)) or None

    result = run_selfplay_games(
        model,
        config,
        num_games=int(games_per_epoch or 0),
        output_dir=Path(ctx.output_dir) / "selfplay",
        epoch=epoch,
        device=device,
        fp16=(device != "cpu"),
        base_seed=base_seed,
        active_games=active,
        virtual_batch_size=vbatch,
    )
    summary = {"status": "completed", **result.as_dict()}
    if hasattr(ctx, "diagnostics"):
        ctx.diagnostics.write_json(f"hexgt.selfplay.epoch_{epoch:06d}.json", summary)
    return summary
