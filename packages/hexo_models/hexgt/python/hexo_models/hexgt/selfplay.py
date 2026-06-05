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

import statistics
from collections import Counter
from dataclasses import dataclass, field, replace
from math import log
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from hexo_models.dense_cnn.compact_io import write_compact_shard
from hexo_models.dense_cnn.replay import materialize_policy_surprise_rows
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
    halflife: float = 0.0,
) -> float:
    """Played-move temperature at ply `move_index` (copy of dense_cnn's rule).

    `halflife` (KataGo-style smooth exponential decay) takes precedence when > 0:
    ``temp = floor + (initial - floor) * 2**(-move_index / halflife)`` — starts at
    `initial`, halves the gap to `floor` every `halflife` plies, asymptotes to `floor`
    (honored late-game floor, never goes greedy). Else `schedule` (sorted
    ``(move, temp)`` anchors) takes precedence: piecewise linear, held at the first
    anchor before it, the final slope continued past the last anchor down to `floor`.
    Otherwise linear decay `initial`->`final` over `decay_moves`, held flat after.
    The opening explores, the endgame sharpens.
    """

    if halflife > 0.0:
        return float(floor + (initial - floor) * (2.0 ** (-float(move_index) / halflife)))
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


_PCR_MASK = (1 << 64) - 1


def _pcr_is_full(
    base_seed: int, epoch: int, game_key: int, move_index: int, full_proportion: float
) -> bool:
    """KataGo Playout Cap Randomization per-move coin (Wu 2020).

    Returns True for a FULL search (large visit cap, recorded as a training row,
    with Dirichlet noise + forced playouts + the temperature schedule) with
    probability ``full_proportion``, else False for a FAST search (small cap, NOT
    recorded, no noise, played greedily). The draw is a deterministic splitmix64
    hash of ``(base_seed, epoch, game_key, move_index)`` so the full/fast schedule
    is reproducible AND decorrelated across games/moves/epochs (the unique move
    coordinate avoids the correlated-noise trap the per-round search seed fixed).
    """

    if full_proportion >= 1.0:
        return True
    if full_proportion <= 0.0:
        return False
    x = (
        (int(base_seed) & _PCR_MASK)
        ^ ((int(epoch) * 0x9E3779B97F4A7C15) & _PCR_MASK)
        ^ ((int(game_key) * 0xC2B2AE3D27D4EB4F) & _PCR_MASK)
        ^ ((int(move_index) * 0x165667B19E3779F9) & _PCR_MASK)
    ) & _PCR_MASK
    # splitmix64 finalizer.
    x = (x + 0x9E3779B97F4A7C15) & _PCR_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _PCR_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _PCR_MASK
    x = (x ^ (x >> 31)) & _PCR_MASK
    unit = (x >> 11) * (1.0 / float(1 << 53))
    return unit < float(full_proportion)


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


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def _counter_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy (nats) of a list of bucket counts."""

    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * log(p)
    return ent


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
    # Q3 + play-style: aggregate signal across all searched positions.
    mean_visit_entropy: float = 0.0
    mean_prior_entropy: float = 0.0
    mean_top_visit_fraction: float = 0.0
    mean_candidate_count: float = 0.0
    # Q1 decisiveness / game length.
    decisive_fraction: float = 0.0
    game_length_median: float = 0.0
    game_length_mean: float = 0.0
    game_length_max: int = 0
    game_length_stdev: float = 0.0
    game_length_p95: float = 0.0
    # Q2 opening / move diversity.
    opening_unique_fraction: float = 0.0
    opening_entropy: float = 0.0
    move2_entropy: float = 0.0
    # Q4 value-target balance.
    win_p0_fraction: float = 0.0
    win_p1_fraction: float = 0.0
    draw_fraction: float = 0.0
    mean_abs_value: float = 0.0
    # Q5 tactical proxy: fraction of decisions with a near-forced best move.
    forced_move_fraction: float = 0.0
    # Non-finite-logit sanitization audit (see HexgtInference._count_and_sanitize).
    # A healthy model should keep these at 0; a non-zero count is a red flag (fp16
    # overflow / model instability) AND drives training exclusion of the affected
    # positions, so it is surfaced in diagnostics rather than left silent.
    sanitized_logit_events: int = 0      # non-finite logit values sanitized this epoch
    sanitized_positions: int = 0         # graphs/leaves with >=1 sanitized logit
    sanitized_search_rounds: int = 0     # search rounds in which >=1 logit was sanitized
    sanitized_samples_excluded: int = 0  # training rows dropped (their search round was tainted)
    # KataGo policy-surprise weighting (row duplication; see materialize_policy_surprise_rows).
    effective_samples: int = 0           # rows actually written (= raw after duplication)
    policy_surprise_mean: float = 0.0    # mean KL(visits||prior) over written positions
    frequency_weight_mean: float = 0.0   # mean per-row duplication weight (1.0 when off)
    # KataGo Playout Cap Randomization (see _pcr_is_full / HEXGT_PCR_KATAGO_MAPPING.md).
    # When OFF, every searched position is a full search and recorded (full==searched,
    # fast==0), so these stay consistent with a non-PCR run.
    pcr_enabled: bool = False
    pcr_full_proportion: float = 0.0
    pcr_full_visits: int = 0             # full-search visit cap (= search_visits)
    pcr_fast_visits: int = 0             # fast-search visit cap
    full_search_count: int = 0           # positions searched at the full cap (recorded)
    fast_search_count: int = 0           # positions searched at the fast cap (NOT recorded)
    recorded_positions: int = 0          # positions that became training rows (== full)
    pcr_fast_rows_excluded: int = 0      # finalized rows dropped because they were fast moves
    mean_search_visits: float = 0.0      # mean visits per searched position (full+fast)

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
            # Q3 / play-style
            "mean_visit_entropy": self.mean_visit_entropy,
            "mean_prior_entropy": self.mean_prior_entropy,
            "mean_top_visit_fraction": self.mean_top_visit_fraction,
            "mean_candidate_count": self.mean_candidate_count,
            # Q1 decisiveness
            "decisive_fraction": self.decisive_fraction,
            "game_length_median": self.game_length_median,
            "game_length_mean": self.game_length_mean,
            "game_length_max": self.game_length_max,
            "game_length_stdev": self.game_length_stdev,
            "game_length_p95": self.game_length_p95,
            # Q2 diversity
            "opening_unique_fraction": self.opening_unique_fraction,
            "opening_entropy": self.opening_entropy,
            "move2_entropy": self.move2_entropy,
            # Q4 value balance
            "win_p0_fraction": self.win_p0_fraction,
            "win_p1_fraction": self.win_p1_fraction,
            "draw_fraction": self.draw_fraction,
            "mean_abs_value": self.mean_abs_value,
            # Q5 tactical proxy
            "forced_move_fraction": self.forced_move_fraction,
            # Sanitization audit
            "sanitized_logit_events": self.sanitized_logit_events,
            "sanitized_positions": self.sanitized_positions,
            "sanitized_search_rounds": self.sanitized_search_rounds,
            "sanitized_samples_excluded": self.sanitized_samples_excluded,
            # Policy-surprise weighting
            "effective_samples": self.effective_samples,
            "policy_surprise_mean": self.policy_surprise_mean,
            "frequency_weight_mean": self.frequency_weight_mean,
            # Playout Cap Randomization
            "pcr_enabled": self.pcr_enabled,
            "pcr_full_proportion": self.pcr_full_proportion,
            "pcr_full_visits": self.pcr_full_visits,
            "pcr_fast_visits": self.pcr_fast_visits,
            "full_search_count": self.full_search_count,
            "fast_search_count": self.fast_search_count,
            "recorded_positions": self.recorded_positions,
            "pcr_fast_rows_excluded": self.pcr_fast_rows_excluded,
            "mean_search_visits": self.mean_search_visits,
        }


# Players recorded in the .hxr game records (P0 opens, P1 second), mirroring
# dense_cnn's selfplay record players.
_RECORD_PLAYERS = (
    HexoRecordPlayer("hexgt-a", "player0", "hexgt A"),
    HexoRecordPlayer("hexgt-b", "player1", "hexgt B"),
)


def _write_game_record(record_file, game, winner, truncated, max_actions):
    """Persist the COMPLETE self-play game as a .hxr record (mirrors dense_cnn /
    the eval harness). `game["actions"]` is the full move sequence INCLUDING the
    terminal/winning move, so the .hxr is the authoritative replay — unlike the
    .npz training shards, which only sample nonterminal positions and therefore
    omit the final move. Truncated games (max_actions) finish as aborted."""
    writer = record_file.begin_game(game["game_id"], seed=game["seed"])
    for action_id in game["actions"]:
        writer.record_action(engine.PlacementAction(unpack_coord_id(int(action_id))))
    if truncated:
        writer.finish_aborted(
            AbortRecord(
                stage="selfplay",
                exception_type="MaxActionsReached",
                message=f"hexgt self-play reached max_actions={max_actions}",
            )
        )
    else:
        writer.finish_completed(winner, len(game["actions"]))


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
    forced_decisions = 0
    # Q-metric accumulators.
    game_lengths: list[int] = []
    openings: list[tuple[int, ...]] = []  # first-6-ply action ids per game
    move2_choices: list[int] = []         # the 2nd-ply action (first real decision)
    win_counts = {"player0": 0, "player1": 0, "draw": 0}
    abs_value_sum = 0.0
    abs_value_n = 0
    # Sanitization-audit accumulators (see HexgtInference._count_and_sanitize).
    sanitized_search_rounds = 0
    sanitized_samples_excluded = 0
    # Policy-surprise (row-duplication) accumulators.
    effective_samples = 0
    surprise_sum = 0.0
    freq_sum = 0.0
    surprise_n = 0
    # KataGo Playout Cap Randomization (see _pcr_is_full / HEXGT_PCR_KATAGO_MAPPING.md).
    # FULL searches use selfplay.search_visits (recorded, noise + forced playouts +
    # temperature); FAST searches use pcr_fast_visits (NOT recorded, no noise, greedy).
    pcr_enabled = bool(selfplay.pcr_enabled)
    pcr_full_proportion = float(selfplay.pcr_full_proportion)
    full_visits = int(selfplay.search_visits)
    fast_visits = int(selfplay.pcr_fast_visits)
    if pcr_enabled:
        if not 0.0 < pcr_full_proportion <= 1.0:
            raise ValueError(
                f"pcr_full_proportion must be in (0, 1], got {pcr_full_proportion!r}"
            )
        if fast_visits < 1:
            raise ValueError(
                f"pcr_fast_visits must be >= 1 when PCR is enabled, got {fast_visits!r}"
            )
    full_search_count = 0
    fast_search_count = 0
    recorded_positions = 0
    pcr_fast_rows_excluded = 0
    next_game_index = 0
    active: list[dict[str, Any]] = []
    started = perf_counter()
    _OPENING_PLIES = 6
    _FORCED_THRESHOLD = 0.9
    # Per-search-round counter feeding a DISTINCT seed into each MCTS batch. Using
    # a fixed `base_seed + epoch` for every round (the old behavior) made the root
    # Dirichlet noise + temperature sampling repeat across rounds — the native
    # session derives per-root randomness from (seed, root_index), so a constant
    # seed gives correlated noise across rounds and identical root noise for a game
    # that keeps its batch index across moves, reducing self-play diversity. A
    # per-round seed decorrelates both. Offset by 100_000 to stay clear of the
    # per-game state-init seed range (base_seed + epoch*1e6 + game_index).
    search_round = 0

    # Full-game .hxr record for the epoch (one file, all games), alongside the
    # .npz training shards. Written incrementally per finished game and closed
    # after the loop. This is the authoritative replay source (the shards omit
    # the terminal/winning move). Mirrors dense_cnn's per-epoch record file.
    record_path = output_dir / f"epoch_{epoch:06d}.hxr"
    record_file = HexoRecordFile.create(record_path, engine.engine_metadata(), _RECORD_PLAYERS)

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
            # KataGo Playout Cap Randomization: assign each playable game's CURRENT
            # move to a FULL or FAST search by a deterministic per-(game, move) coin.
            # FULL == search_visits, recorded, with Dirichlet noise + forced playouts +
            # the temperature schedule. FAST == pcr_fast_visits, NOT recorded, no noise,
            # no forced playouts, played greedily. Each subset is searched in its own
            # batched run() call (leaves still coalesce within the subset), preserving
            # the batched-throughput property. When PCR is off there is one subset (all
            # full), identical to the pre-PCR single call. See HEXGT_PCR_KATAGO_MAPPING.md.
            if pcr_enabled:
                full_games: list[dict[str, Any]] = []
                fast_games: list[dict[str, Any]] = []
                for game in playable:
                    bucket = (
                        full_games
                        if _pcr_is_full(
                            base_seed, epoch, int(game["search_key"]),
                            len(game["actions"]), pcr_full_proportion,
                        )
                        else fast_games
                    )
                    bucket.append(game)
                search_specs = [(full_games, True), (fast_games, False)]
            else:
                search_specs = [(playable, True)]
            round_seed_base = base_seed + epoch * 1_000_000 + 100_000 + search_round
            for spec_index, (spec_games, is_full) in enumerate(search_specs):
                if not spec_games:
                    continue
                spec_visits = full_visits if is_full else fast_visits
                if is_full:
                    # Full searches carry the exploration machinery (KataGo: full only).
                    move_temperatures = [
                        _move_temperature(
                            len(game["actions"]),
                            initial=selfplay.temperature,
                            final=selfplay.final_temperature,
                            decay_moves=selfplay.temperature_decay_moves,
                            schedule=selfplay.temperature_schedule,
                            floor=selfplay.temperature_floor,
                            halflife=selfplay.temperature_halflife,
                        )
                        for game in spec_games
                    ]
                    spec_alpha = (
                        selfplay.root_dirichlet_total_alpha
                        if selfplay.root_dirichlet_noise_enabled
                        else None
                    )
                    spec_fraction = (
                        selfplay.root_dirichlet_noise_fraction
                        if selfplay.root_dirichlet_noise_enabled
                        else None
                    )
                    spec_forced_k = selfplay.forced_playout_k
                else:
                    # Fast searches run clean: no Dirichlet noise, no forced playouts,
                    # greedy move selection (temperature 0 -> argmax of the visit policy).
                    move_temperatures = [0.0] * len(spec_games)
                    spec_alpha = None
                    spec_fraction = None
                    spec_forced_k = 0.0
                # Distinct seed per subset so full/fast noise + sampling stay decorrelated.
                spec_seed = round_seed_base + spec_index * 0x5DEECE66D
                search_started = perf_counter()
                sanitize_before = inference.sanitized_logit_total
                searches = mcts_session.run(
                    [game["search_key"] for game in spec_games],
                    [game["state"] for game in spec_games],
                    inference,
                    visits=spec_visits,
                    c_puct=selfplay.c_puct,
                    temperature=selfplay.temperature,
                    seed=spec_seed,
                    virtual_batch_size=vbatch,
                    active_root_limit=selfplay.mcts_active_root_limit,
                    root_dirichlet_total_alpha=spec_alpha,
                    root_dirichlet_noise_fraction=spec_fraction,
                    root_policy_temperature=selfplay.root_policy_temperature,
                    fpu_reduction=selfplay.fpu_reduction,
                    virtual_loss=selfplay.virtual_loss,
                    widening_policy_mass=selfplay.widening_policy_mass,
                    widening_max_children=selfplay.widening_max_children,
                    widening_min_children=selfplay.widening_min_children,
                    forced_playout_k=spec_forced_k,
                    move_temperatures=move_temperatures,
                )
                mcts_search_elapsed += perf_counter() - search_started
                # Conservative round-level sanitization taint: the native session
                # coalesces the in-flight leaves of ALL concurrent games into one batched
                # eval, so Python cannot attribute a sanitized leaf to a single game/
                # position (that needs a Rust-side per-leaf signal). A sanitized leaf still
                # backs up into the per-game root statistics, so when ANY logit was
                # sanitized this subset's run we flag EVERY position decided in it and
                # exclude them at write time. Rare in a healthy model; counters surface it.
                round_tainted = inference.sanitized_logit_total > sanitize_before
                if round_tainted:
                    sanitized_search_rounds += 1
                if len(searches) != len(spec_games):
                    raise RuntimeError(
                        f"hexgt MCTS returned {len(searches)} results for {len(spec_games)} playable games"
                    )
                for game, search, temp in zip(spec_games, searches, move_temperatures):
                    searched_positions += 1
                    mcts_simulations += int(search.visits)
                    if is_full:
                        full_search_count += 1
                        recorded_positions += 1
                    else:
                        fast_search_count += 1
                    state = game["state"]
                    visit_pairs = list(search.visit_policy)
                    prior_pairs = list(search.root_prior_policy)
                    v_ent, v_top, v_support = _policy_entropy(visit_pairs)
                    p_ent, _p_top, _p_support = _policy_entropy(prior_pairs)
                    # Play-style / data-quality aggregates describe the RECORDED (full)
                    # positions only, so they stay comparable to a non-PCR run (where
                    # every searched position is recorded). Fast moves are counted but
                    # not folded into these means.
                    if is_full:
                        ent_visit_sum += v_ent
                        ent_prior_sum += p_ent
                        top_visit_sum += v_top
                        cand_count_sum += len(prior_pairs)
                        if v_top >= _FORCED_THRESHOLD:
                            forced_decisions += 1
                    sample = sample_from_state(
                        state,
                        game_id=game["game_id"],
                        turn_index=len(game["actions"]),
                        policy=visit_pairs,
                        root_prior_policy=prior_pairs,
                        metadata={"epoch": epoch, "search_visits": search.visits},
                    )
                    # Tag the PCR class so the write step keeps only full-search rows as
                    # training data (KataGo records full only). Fast rows still feed the
                    # DENSE STV/opp-policy aux chain in `pending` (Option A), then drop.
                    sample_meta = {**dict(sample.metadata), "pcr_full": bool(is_full)}
                    if round_tainted:
                        # Flag so finalize/write can exclude this position from training
                        # targets (a sanitized neutral output must not become a target).
                        sample_meta["search_sanitized"] = True
                    sample = replace(sample, metadata=sample_meta)
                    game["pending"].append((sample.current_player, sample, search.root_value))
                    if game["trace"] is not None:
                        game["trace"].append(
                            {
                                "move": len(game["actions"]),
                                "player": str(sample.current_player),
                                "action_id": int(search.action_id),
                                "root_value": float(search.root_value),
                                "visits": int(search.visits),
                                "pcr_full": bool(is_full),
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
            search_round += 1

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
            # Persist the full game record (incl. the terminal/winning move).
            _write_game_record(record_file, game, winner, truncated, selfplay.max_actions)
            # Finalize over the DENSE pending trajectory (full AND fast moves) so the
            # STV (EMA-of-future-root-value) and opp-policy aux targets keep their
            # per-ply horizon semantics; PCR fast rows are dropped below, not here.
            finalized = finalize_game_samples(
                game["pending"],
                winner,
                horizons,
                truncated=truncated,
                soft_z_lambda=config.samples.soft_z_lambda,
                # PCR: only train the opponent-policy head on full->full transitions
                # (mask the target when the next opponent move was a fast search).
                mask_opp_from_fast=pcr_enabled,
            )
            # EXCLUDE positions whose search round was sanitization-tainted: a
            # near-uniform prior + neutral value produced by the fp16-overflow guard
            # is not a trustworthy training target. The drop count is surfaced.
            clean = [s for s in finalized if not s.metadata.get("search_sanitized")]
            sanitized_samples_excluded += len(finalized) - len(clean)
            # KataGo Playout Cap Randomization: record ONLY full-search positions as
            # training rows. Fast moves advanced the game and fed the dense STV/opp
            # chain above, but never become policy/value targets. `pcr_full` defaults
            # True so a non-PCR shard (flag always True) keeps every row unchanged.
            to_write = [s for s in clean if s.metadata.get("pcr_full", True)]
            pcr_fast_rows_excluded += len(clean) - len(to_write)
            # KataGo policy-surprise weighting: repeat each position by a frequency
            # weight driven by KL(visit-policy || root-prior), so positions where the
            # search most disagreed with the network prior are trained on more. The
            # root prior (which compact_io drops at write time) is still present on the
            # finalized samples HERE, so the KL is exact. Off -> rows written 1:1.
            if config.samples.policy_surprise_enabled and to_write:
                materialized, weight_stats = materialize_policy_surprise_rows(
                    to_write,
                    seed=base_seed + epoch * 1_000_000_003 + int(game["search_key"]),
                    uniform_fraction=config.samples.policy_surprise_uniform_fraction,
                    max_weight=config.samples.policy_surprise_max_weight,
                )
                surprise_sum += weight_stats["policy_surprise_mean"] * len(to_write)
                freq_sum += weight_stats["frequency_weight_mean"] * len(to_write)
                surprise_n += len(to_write)
            else:
                materialized = to_write
            npz_path = output_dir / f"epoch_{epoch:06d}_game_{int(game['search_key']):06d}.npz"
            write_compact_shard(npz_path, materialized, short_term_value_horizons=horizons)
            shard_paths.append(str(npz_path))
            raw_samples += len(to_write)
            effective_samples += len(materialized)
            # Q1 length, Q2 diversity, Q4 value balance.
            game_lengths.append(len(game["actions"]))
            openings.append(tuple(int(a) for a in game["actions"][:_OPENING_PLIES]))
            if len(game["actions"]) >= 2:
                move2_choices.append(int(game["actions"][1]))
            if truncated:
                win_counts["draw"] += 1
            elif winner in win_counts:
                win_counts[winner] += 1
            else:
                win_counts["draw"] += 1
            # |value| over the RECORDED rows (the training targets), so it reflects the
            # data the model actually learns from (and stays comparable under PCR).
            for fs in to_write:
                abs_value_sum += abs(float(fs.value))
                abs_value_n += 1
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

    record_file.close()

    elapsed = perf_counter() - started
    sp = max(1, searched_positions)
    # Play-style / data-quality means are accumulated over RECORDED (full-search)
    # positions only, so divide by that count (== searched when PCR is off).
    rp = max(1, recorded_positions)
    n_games = max(1, completed_games + truncated_games)
    opening_counts = Counter(openings)
    move2_counts = Counter(move2_choices)
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
        mean_visit_entropy=ent_visit_sum / rp,
        mean_prior_entropy=ent_prior_sum / rp,
        mean_top_visit_fraction=top_visit_sum / rp,
        mean_candidate_count=cand_count_sum / rp,
        decisive_fraction=completed_games / n_games,
        game_length_median=_percentile(game_lengths, 50),
        game_length_mean=(sum(game_lengths) / len(game_lengths)) if game_lengths else 0.0,
        game_length_max=max(game_lengths) if game_lengths else 0,
        game_length_stdev=(statistics.pstdev(game_lengths) if len(game_lengths) > 1 else 0.0),
        game_length_p95=_percentile(game_lengths, 95),
        opening_unique_fraction=len(opening_counts) / n_games,
        opening_entropy=_counter_entropy(list(opening_counts.values())),
        move2_entropy=_counter_entropy(list(move2_counts.values())),
        win_p0_fraction=win_counts["player0"] / n_games,
        win_p1_fraction=win_counts["player1"] / n_games,
        draw_fraction=win_counts["draw"] / n_games,
        mean_abs_value=abs_value_sum / max(1, abs_value_n),
        forced_move_fraction=forced_decisions / rp,
        sanitized_logit_events=int(inference.sanitized_logit_total),
        sanitized_positions=int(inference.sanitized_graph_total),
        sanitized_search_rounds=int(sanitized_search_rounds),
        sanitized_samples_excluded=int(sanitized_samples_excluded),
        effective_samples=int(effective_samples),
        policy_surprise_mean=(surprise_sum / surprise_n) if surprise_n else 0.0,
        frequency_weight_mean=(freq_sum / surprise_n) if surprise_n else 0.0,
        pcr_enabled=pcr_enabled,
        pcr_full_proportion=(pcr_full_proportion if pcr_enabled else 0.0),
        pcr_full_visits=full_visits,
        pcr_fast_visits=(fast_visits if pcr_enabled else 0),
        full_search_count=int(full_search_count),
        fast_search_count=int(fast_search_count),
        recorded_positions=int(recorded_positions),
        pcr_fast_rows_excluded=int(pcr_fast_rows_excluded),
        mean_search_visits=(mcts_simulations / sp),
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
