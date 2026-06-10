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

import math
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


def _root_policy_temperature(
    move_index: int, *, base: float, early: float, halflife_plies: float
) -> float:
    """KataGo root-policy temperature early ramp (interpolateEarly).

    `T(ply) = base + (early - base) * 0.5 ** (ply / halflife)` — KataGo's
    self-play search runs rootPolicyTemperatureEarly (1.25) at the opening,
    decaying exponentially to rootPolicyTemperature (1.1). `early <= 0` disables
    the ramp (constant `base`).
    """

    if early <= 0.0 or halflife_plies <= 0.0:
        return base
    return base + (early - base) * 0.5 ** (move_index / halflife_plies)


_PCR_MASK = (1 << 64) - 1


def _splitmix64_unit(*parts: int) -> float:
    """Deterministic uniform in [0, 1) from a splitmix64 finalizer over xor-folded
    odd-multiplied parts (the hexgt `_pcr_is_full` construction). Decorrelated
    across every distinct (part...) coordinate; reproducible per run seed."""

    multipliers = (
        0x9E3779B97F4A7C15,
        0xC2B2AE3D27D4EB4F,
        0x165667B19E3779F9,
        0xD6E8FEB86659FD93,
    )
    x = 0
    for index, part in enumerate(parts):
        multiplier = multipliers[index % len(multipliers)] | 1
        x ^= (int(part) * multiplier) & _PCR_MASK
    x = (x + 0x9E3779B97F4A7C15) & _PCR_MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _PCR_MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _PCR_MASK
    x = (x ^ (x >> 31)) & _PCR_MASK
    return (x >> 11) * (1.0 / float(1 << 53))


def _pcr_is_full(
    base_seed: int, epoch: int, game_key: int, move_index: int, full_proportion: float
) -> bool:
    """KataGo Playout Cap Randomization per-move coin (Wu 2020).

    True for a FULL search (large visit cap, recorded as a training row, with
    Dirichlet noise + forced playouts + the temperature schedule) with
    probability ``full_proportion``, else False for a FAST search (small cap,
    NOT recorded, no noise, played greedily). Deterministic in
    (base_seed, epoch, game_key, move_index) — the unique move coordinate keeps
    the full/fast schedule decorrelated across games/moves/epochs.
    """

    if full_proportion >= 1.0:
        return True
    if full_proportion <= 0.0:
        return False
    return _splitmix64_unit(base_seed, epoch * 3 + 1, game_key, move_index) < float(full_proportion)


def _policy_init_plies(
    base_seed: int,
    epoch: int,
    game_key: int,
    *,
    fraction: float,
    avg_plies: float,
    max_plies: int,
) -> int:
    """Number of policy-initialized opening plies for one game (KataGo
    initGamesWithPolicy). With probability `fraction` the game is selected and
    the ply count is drawn from a truncated exponential with mean `avg_plies`,
    capped at `max_plies` (a draw of 0 is possible, as in KataGo). Deterministic
    per (base_seed, epoch, game_key)."""

    if fraction <= 0.0 or avg_plies <= 0.0 or max_plies <= 0:
        return 0
    if fraction < 1.0 and _splitmix64_unit(base_seed, epoch * 5 + 2, game_key, 1) >= fraction:
        return 0
    unit = _splitmix64_unit(base_seed, epoch * 7 + 3, game_key, 2)
    count = int(-float(avg_plies) * math.log(max(1.0e-12, 1.0 - unit)))
    return min(int(max_plies), count)


def _sample_policy_init_action(
    prior_pairs: Any, temperature: float, *, base_seed: int, epoch: int, game_key: int, move_index: int
) -> int:
    """Sample an action id from the raw root prior at `prior ** (1/T)`.

    `prior_pairs` is the search payload's `root_prior_policy` — the normalized
    union of every in-crop legal move (the policy-init subset runs with noise off
    and root temperature 1.0, so this is exactly the raw net prior). Deterministic
    per (base_seed, epoch, game_key, move_index)."""

    pairs = [(int(action_id), float(weight)) for action_id, weight in prior_pairs if float(weight) > 0.0]
    if not pairs:
        raise RuntimeError("policy-init sampling received an empty root prior")
    if temperature > 0.0 and abs(temperature - 1.0) > 1.0e-9:
        inverse = 1.0 / float(temperature)
        # Normalize by the max weight before powering so extreme temperatures
        # stay finite (max term is exactly 1.0; the rest underflow toward 0).
        max_weight = max(weight for _action_id, weight in pairs)
        pairs = [(action_id, (weight / max_weight) ** inverse) for action_id, weight in pairs]
    total = sum(weight for _action_id, weight in pairs)
    if not total > 0.0:
        raise RuntimeError("policy-init sampling prior mass is not positive")
    unit = _splitmix64_unit(base_seed, epoch * 11 + 5, game_key, move_index) * total
    cumulative = 0.0
    for action_id, weight in pairs:
        cumulative += weight
        if unit < cumulative:
            return action_id
    return pairs[-1][0]


def _validate_selfplay_levers(selfplay: Any) -> tuple[bool, bool, float, float]:
    """Validate the PCR / policy-init / root-temp-ramp knobs (both schedulers).

    Returns (pcr_enabled, policy_init_enabled, root_temp_early, root_temp_halflife).
    """

    pcr_enabled = bool(selfplay.pcr_enabled)
    if pcr_enabled:
        proportion = float(selfplay.pcr_full_proportion)
        if not 0.0 < proportion <= 1.0:
            raise ValueError(f"pcr_full_proportion must be in (0, 1], got {proportion!r}")
        if int(selfplay.pcr_fast_visits) < 1:
            raise ValueError(
                f"pcr_fast_visits must be >= 1 when PCR is enabled, got {selfplay.pcr_fast_visits!r}"
            )
    policy_init_enabled = float(selfplay.policy_init_fraction) > 0.0
    if policy_init_enabled:
        if not 0.0 < float(selfplay.policy_init_fraction) <= 1.0:
            raise ValueError(
                f"policy_init_fraction must be in (0, 1], got {selfplay.policy_init_fraction!r}"
            )
        if float(selfplay.policy_init_avg_plies) <= 0.0 or int(selfplay.policy_init_max_plies) < 1:
            raise ValueError(
                "policy_init_avg_plies must be > 0 and policy_init_max_plies >= 1 "
                "when policy-initialized openings are enabled"
            )
        if not float(selfplay.policy_init_temperature) > 0.0:
            raise ValueError(
                f"policy_init_temperature must be > 0, got {selfplay.policy_init_temperature!r}"
            )
    root_temp_early = float(selfplay.root_policy_temperature_early)
    root_temp_halflife = float(selfplay.root_policy_temperature_halflife)
    if root_temp_early > 0.0 and root_temp_halflife <= 0.0:
        raise ValueError(
            "root_policy_temperature_halflife must be > 0 when root_policy_temperature_early is set"
        )
    return pcr_enabled, policy_init_enabled, root_temp_early, root_temp_halflife


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
    adaptive_vbatch = _adaptive_vbatch_enabled() or bool(selfplay.adaptive_virtual_batch)
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

    # KataGo Playout Cap Randomization + policy-initialized openings + the
    # root-policy temperature early ramp (see _pcr_is_full / _policy_init_plies /
    # _root_policy_temperature and the hexgt lineage's HEXGT_PCR_KATAGO_MAPPING.md).
    # FULL searches carry the exploration machinery and become training rows;
    # FAST/INIT moves advance the game cheaply and are dropped at write time.
    pcr_enabled, policy_init_enabled, root_temp_early, root_temp_halflife = (
        _validate_selfplay_levers(selfplay)
    )
    pcr_full_proportion = float(selfplay.pcr_full_proportion)
    pcr_fast_visits = int(selfplay.pcr_fast_visits)

    samples_added = 0
    raw_samples_added = 0
    total_decisions = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    mcts_search_elapsed = 0.0
    full_search_count = 0
    fast_search_count = 0
    policy_init_moves = 0
    pcr_fast_rows_excluded = 0
    npz_skipped_empty = 0
    # Observation-only spill telemetry: per-category facts beyond hex distance 20
    # from the crop center, which the fixed radius-20 disk crop cannot represent
    # (Spec A). Accumulated over every RECORDED (full-search) position's sample so
    # the telemetry stays comparable to a non-PCR run.
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
                        # Policy-initialized opening: plies < this count are
                        # sampled from the raw prior instead of searched.
                        "policy_init_plies": _policy_init_plies(
                            base_seed,
                            epoch,
                            next_game_index,
                            fraction=float(selfplay.policy_init_fraction),
                            avg_plies=float(selfplay.policy_init_avg_plies),
                            max_plies=int(selfplay.policy_init_max_plies),
                        )
                        if policy_init_enabled
                        else 0,
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
                # Split this round's playable games into search subsets:
                # - "init": policy-initialized opening plies — a 1-visit probe
                #   (noise off, root temperature 1.0) whose exported root prior IS
                #   the raw net prior; the played action is sampled from it in
                #   Python and the position is NOT recorded as a training row.
                # - "full": the normal recorded search (noise + forced playouts +
                #   temperature schedule).
                # - "fast": PCR fast search — cheap visit cap, no noise, no forced
                #   playouts, greedy selection, NOT recorded.
                # Each subset runs as its own batched run() call (leaves still
                # coalesce within the subset); with PCR and policy-init disabled
                # there is one subset (all full), identical to the previous
                # single-call behavior.
                init_games: list[dict[str, Any]] = []
                full_games: list[dict[str, Any]] = []
                fast_games: list[dict[str, Any]] = []
                for game in playable:
                    if policy_init_enabled and len(game["actions"]) < int(game["policy_init_plies"]):
                        init_games.append(game)
                    elif pcr_enabled and not _pcr_is_full(
                        base_seed,
                        epoch,
                        int(game["search_key"]),
                        len(game["actions"]),
                        pcr_full_proportion,
                    ):
                        fast_games.append(game)
                    else:
                        full_games.append(game)
                search_specs = [
                    (full_games, "full"),
                    (fast_games, "fast"),
                    (init_games, "init"),
                ]
                round_seed_base = base_seed + epoch * 1_000_003 + search_rounds
                for spec_index, (spec_games, spec_kind) in enumerate(search_specs):
                    if not spec_games:
                        continue
                    if spec_kind == "full":
                        spec_visits = selfplay.search_visits
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
                            for game in spec_games
                        ]
                        # KataGo root-policy temperature early ramp; full
                        # searches only (fast/init run exploitation-clean).
                        root_policy_temperatures = [
                            _root_policy_temperature(
                                len(game["actions"]),
                                base=selfplay.root_policy_temperature,
                                early=root_temp_early,
                                halflife_plies=root_temp_halflife,
                            )
                            for game in spec_games
                        ]
                    elif spec_kind == "fast":
                        spec_visits = pcr_fast_visits
                        spec_alpha = None
                        spec_fraction = None
                        spec_forced_k = 0.0
                        # Greedy (argmax of visits), no exploration machinery.
                        move_temperatures = [0.0] * len(spec_games)
                        root_policy_temperatures = [1.0] * len(spec_games)
                    else:  # "init"
                        spec_visits = 1
                        spec_alpha = None
                        spec_fraction = None
                        spec_forced_k = 0.0
                        # The selected action is ignored (Python samples from the
                        # exported raw prior below); temperature 1.0 keeps the
                        # exported root prior untransformed.
                        move_temperatures = [1.0] * len(spec_games)
                        root_policy_temperatures = [1.0] * len(spec_games)
                    # Distinct seed per subset so full/fast/init noise + sampling
                    # stay decorrelated within the round.
                    spec_seed = round_seed_base + spec_index * 0x5DEECE66D
                    # Adaptive virtual_batch_size, PER SUBSET: hold the round's
                    # leaf-request budget (~active_limit * base_vbatch) constant
                    # across this subset's games, so a small subset (e.g. the
                    # ~25% full-search games under PCR, or the end-of-epoch
                    # drain tail) still fills the calibrated inference batch.
                    # Bounded by the subset's visit cap. Costs a little search
                    # quality (higher vbatch -> more virtual-loss-correlated
                    # selection) on the affected searches.
                    spec_vbatch = trainer.mcts_virtual_batch_size
                    if adaptive_vbatch:
                        budget = active_limit * trainer.mcts_virtual_batch_size
                        spec_vbatch = max(
                            trainer.mcts_virtual_batch_size,
                            min(int(spec_visits), -(-budget // len(spec_games))),
                        )
                    searches = mcts_session.run(
                        [game["search_key"] for game in spec_games],
                        [game["state"] for game in spec_games],
                        inference,
                        visits=spec_visits,
                        c_puct=selfplay.c_puct,
                        temperature=selfplay.temperature,
                        seed=spec_seed,
                        virtual_batch_size=spec_vbatch,
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
                        root_policy_temperatures=root_policy_temperatures,
                    )
                    if len(searches) != len(spec_games):
                        raise RuntimeError(
                            f"dense_cnn MCTS returned {len(searches)} results for {len(spec_games)} playable games"
                        )
                    _extend_mcts_diagnostic_batches(mcts_diagnostic_batches, searches)
                    for game, search in zip(spec_games, searches):
                        if int(search.visits) != spec_visits:
                            raise RuntimeError(
                                f"dense_cnn MCTS returned {search.visits} visits; expected exactly {spec_visits}"
                            )
                        state = game["state"]
                        if spec_kind == "init":
                            # KataGo initGamesWithPolicy: sample the played move
                            # from the raw prior; record nothing (no search
                            # target exists). The pending chain gets a masked
                            # placeholder row so the dense STV / opp-policy /
                            # moves-left target indexing stays per-ply exact;
                            # root_value is the 1-visit eval value.
                            policy_init_moves += 1
                            action_id = _sample_policy_init_action(
                                search.root_prior_policy,
                                float(selfplay.policy_init_temperature),
                                base_seed=base_seed,
                                epoch=epoch,
                                game_key=int(game["search_key"]),
                                move_index=len(game["actions"]),
                            )
                            sample = sample_from_state(
                                state,
                                game_id=game["game_id"],
                                turn_index=len(game["actions"]),
                                policy=search.visit_policy,
                                root_prior_policy=search.root_prior_policy,
                                metadata={
                                    "epoch": epoch,
                                    "search_visits": search.visits,
                                    "pcr_full": False,
                                    "policy_init": True,
                                },
                            )
                            game["pending"].append((sample.current_player, sample, search.root_value))
                            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))
                            game["actions"].append(action_id)
                            # The 1-visit probe advanced its native root by ITS
                            # selected action, which is generally not the sampled
                            # one; drop the stale tree so the next round rebuilds
                            # from the live state instead of failing the hash check.
                            mcts_session.discard(int(game["search_key"]))
                            continue
                        searched_positions += 1
                        mcts_simulations += int(search.visits)
                        is_full = spec_kind == "full"
                        if is_full:
                            full_search_count += 1
                        else:
                            fast_search_count += 1
                        # The sample is captured before the chosen action mutates the
                        # state: policy/legal describe the decision position; outcome
                        # targets are filled once the game ends. Fast rows feed the
                        # dense pending chain only (dropped at write time).
                        sample = sample_from_state(
                            state,
                            game_id=game["game_id"],
                            turn_index=len(game["actions"]),
                            policy=search.visit_policy,
                            root_prior_policy=search.root_prior_policy,
                            metadata={
                                "epoch": epoch,
                                "search_visits": search.visits,
                                "pcr_full": is_full,
                            },
                        )
                        if is_full:
                            for category, count in count_spill(sample).items():
                                epoch_spill[category] += count
                        game["pending"].append((sample.current_player, sample, search.root_value))
                        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(search.action_id)))
                        game["actions"].append(search.action_id)
                mcts_search_elapsed += perf_counter() - search_started
                search_rounds += 1

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

                # Finalize over the DENSE pending trajectory (full AND fast/init
                # placeholder rows) so the STV (EMA-of-future-root-value),
                # opp-policy, and moves-left targets keep their per-ply
                # semantics; non-full rows are dropped below, not here.
                finalized = _finalize_game_samples(
                    game["pending"],
                    winner,
                    horizons,
                    truncated=truncated,
                    soft_z_lambda=config.samples.soft_z_lambda,
                    # Only train the opponent-policy head on full->full
                    # transitions (mask the target when the next opponent move
                    # was a fast/policy-init row).
                    mask_opp_from_fast=(pcr_enabled or policy_init_enabled),
                )
                total_decisions += len(finalized)
                # Record ONLY full-search positions as training rows (KataGo
                # records full only). Fast/init moves advanced the game and fed
                # the dense target chain above, but never become policy/value
                # targets. `pcr_full` defaults True so a non-PCR run keeps every
                # row unchanged.
                to_write = [s for s in finalized if s.metadata.get("pcr_full", True)]
                pcr_fast_rows_excluded += len(finalized) - len(to_write)
                if to_write:
                    materialized, weight_stats = materialize_policy_surprise_rows(
                        to_write,
                        seed=base_seed + epoch * 1_000_000_003 + int(game["search_key"]),
                        uniform_fraction=config.samples.policy_surprise_uniform_fraction,
                        max_weight=config.samples.policy_surprise_max_weight,
                    )
                    npz_path = record_dir / f"epoch_{epoch:06d}_game_{int(game['search_key']):06d}.npz"
                    write_result = write_selfplay_npz(
                        npz_path,
                        materialized,
                        raw_rows=len(to_write),
                        epoch=epoch,
                        game_id=str(game["game_id"]),
                        short_term_value_horizons=horizons,
                    )
                    raw_samples_added += len(to_write)
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
                else:
                    # Possible under PCR: a short game whose every move drew the
                    # fast coin. Nothing to train on; skip the shard.
                    npz_skipped_empty += 1
                active.remove(game)
                mcts_session.discard(int(game["search_key"]))

            if perf_counter() - last_live_write >= _LIVE_PROGRESS_INTERVAL_SECONDS:
                _write_live_progress("running")
                last_live_write = perf_counter()

    elapsed = perf_counter() - started
    games_done = completed_games + truncated_games
    if games_done > 0:
        # Game length in DECISIONS (incl. fast/policy-init plies), NOT recorded
        # training rows — the adaptive temperature half-life keys off real plies.
        latest_mean_length = total_decisions / games_done
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
        "total_decisions": total_decisions,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "search_visits": selfplay.search_visits,
        "pcr": {
            "enabled": pcr_enabled,
            "full_proportion": pcr_full_proportion if pcr_enabled else 1.0,
            "fast_visits": pcr_fast_visits if pcr_enabled else 0,
            "full_search_count": full_search_count,
            "fast_search_count": fast_search_count,
            "fast_rows_excluded": pcr_fast_rows_excluded,
            "npz_skipped_empty": npz_skipped_empty,
        },
        "policy_init": {
            "enabled": policy_init_enabled,
            "fraction": float(selfplay.policy_init_fraction),
            "avg_plies": float(selfplay.policy_init_avg_plies),
            "max_plies": int(selfplay.policy_init_max_plies),
            "temperature": float(selfplay.policy_init_temperature),
            "moves": policy_init_moves,
        },
        "root_policy_temperature_control": {
            "base": selfplay.root_policy_temperature,
            "early": root_temp_early,
            "halflife_plies": root_temp_halflife,
        },
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
    # PCR / policy-initialized openings / root-policy temperature ramp: the
    # continuous scheduler implements these natively in Rust (per-slot move
    # classes); validate the knobs here and pass them through. The per-game
    # policy-init draw and the per-move PCR coin live Rust-side (deterministic
    # in base_seed via dedicated mix_seed streams), so the driver only reads the
    # class flags off each move payload.
    pcr_enabled, policy_init_enabled, root_temp_early, root_temp_halflife = (
        _validate_selfplay_levers(selfplay)
    )
    pcr_full_proportion = float(selfplay.pcr_full_proportion)
    pcr_fast_visits = int(selfplay.pcr_fast_visits)

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
    total_decisions = 0
    searched_positions = 0
    mcts_simulations = 0
    games_started = 0
    completed_games = 0
    truncated_games = 0
    full_search_count = 0
    fast_search_count = 0
    policy_init_moves = 0
    pcr_fast_rows_excluded = 0
    npz_skipped_empty = 0
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
                    # Finalize over the DENSE pending trajectory (full AND
                    # fast/init placeholder rows) so the STV / opp-policy /
                    # moves-left targets keep their per-ply semantics; non-full
                    # rows are dropped below, not here.
                    finalized = _finalize_game_samples(
                        game["pending"],
                        winner,
                        horizons,
                        truncated=truncated,
                        soft_z_lambda=config.samples.soft_z_lambda,
                        mask_opp_from_fast=(pcr_enabled or policy_init_enabled),
                    )
                    to_write = [s for s in finalized if s.metadata.get("pcr_full", True)]
                    if to_write:
                        materialized, weight_stats = materialize_policy_surprise_rows(
                            to_write,
                            seed=base_seed + epoch * 1_000_000_003 + int(game["search_key"]),
                            uniform_fraction=config.samples.policy_surprise_uniform_fraction,
                            max_weight=config.samples.policy_surprise_max_weight,
                        )
                        npz_path = record_dir / f"epoch_{epoch:06d}_game_{int(game['search_key']):06d}.npz"
                        write_result = write_selfplay_npz(
                            npz_path,
                            materialized,
                            raw_rows=len(to_write),
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
                                "decisions": len(finalized),
                                "rows_excluded": len(finalized) - len(to_write),
                            }
                        )
                    else:
                        # Possible under PCR: a short game whose every move drew
                        # the fast coin. Nothing to train on; skip the shard.
                        writer_results.append(
                            {
                                "path": None,
                                "raw_rows": 0,
                                "effective_rows": 0,
                                "policy_surprise_mean": 0.0,
                                "frequency_weight_mean": 0.0,
                                "decisions": len(finalized),
                                "rows_excluded": len(finalized),
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
        "full_moves": 0,
        "fast_moves": 0,
        "init_moves": 0,
        "flush_size_histogram": {},
        "on_move_seconds": 0.0,
    }

    with HexoRecordFile.create(record_path, engine.engine_metadata(), players) as record_file:
        writer_thread = threading.Thread(target=_writer, args=(record_file,), name="dense-cnn-continuous-writer")
        writer_thread.start()

        def _on_move(game_key: int, payload: Mapping[str, Any]) -> object:
            nonlocal searched_positions, mcts_simulations, completed_games, truncated_games
            nonlocal next_game_index, games_started, last_live_write
            nonlocal full_search_count, fast_search_count, policy_init_moves
            game = active.pop(int(game_key), None)
            if game is None:
                raise RuntimeError(f"continuous MCTS callback received unknown game key {game_key}")
            if perf_counter() - last_live_write >= _LIVE_PROGRESS_INTERVAL_SECONDS:
                _write_live_progress("running")
                last_live_write = perf_counter()
            search = _result_from_payload(payload)
            # Move class flags set by the Rust scheduler (PCR / policy-init).
            is_init = bool(payload.get("policy_init", False))
            is_full = bool(payload.get("pcr_full", True)) and not is_init
            if is_init:
                expected_visits = 1
            elif is_full:
                expected_visits = selfplay.search_visits
            else:
                expected_visits = pcr_fast_visits
            if int(search.visits) != expected_visits:
                raise RuntimeError(
                    f"dense_cnn MCTS returned {search.visits} visits; expected exactly "
                    f"{expected_visits} ({'init' if is_init else 'full' if is_full else 'fast'})"
                )
            if is_init:
                policy_init_moves += 1
            else:
                searched_positions += 1
                mcts_simulations += int(search.visits)
                if is_full:
                    full_search_count += 1
                else:
                    fast_search_count += 1
            state = game["state"]
            # Fast/init rows feed the dense pending chain only (dropped at write
            # time); the `pcr_full` tag drives that and the opp-policy masking.
            metadata: dict[str, Any] = {
                "epoch": epoch,
                "search_visits": search.visits,
                "pcr_full": is_full,
            }
            if is_init:
                metadata["policy_init"] = True
            sample = sample_from_state(
                state,
                game_id=game["game_id"],
                turn_index=len(game["actions"]),
                policy=search.visit_policy,
                root_prior_policy=search.root_prior_policy,
                metadata=metadata,
            )
            if is_full:
                # Spill telemetry describes RECORDED rows only (comparable to a
                # non-PCR run).
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
                    root_policy_temperature_early=(root_temp_early if root_temp_early > 0.0 else None),
                    root_policy_temperature_halflife=(
                        root_temp_halflife if root_temp_early > 0.0 else None
                    ),
                    pcr_full_proportion=(pcr_full_proportion if pcr_enabled else None),
                    pcr_fast_visits=(pcr_fast_visits if pcr_enabled else None),
                    policy_init_fraction=(
                        float(selfplay.policy_init_fraction) if policy_init_enabled else None
                    ),
                    policy_init_avg_plies=(
                        float(selfplay.policy_init_avg_plies) if policy_init_enabled else None
                    ),
                    policy_init_max_plies=(
                        int(selfplay.policy_init_max_plies) if policy_init_enabled else None
                    ),
                    policy_init_temperature=(
                        float(selfplay.policy_init_temperature) if policy_init_enabled else None
                    ),
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

    # Writer results carry per-game decision counts (the dense trajectory incl.
    # fast/init plies) alongside the recorded-row counts; shards skipped for
    # having no recordable rows have path=None.
    total_decisions = sum(int(item.get("decisions", item["raw_rows"])) for item in writer_results)
    pcr_fast_rows_excluded = sum(int(item.get("rows_excluded", 0)) for item in writer_results)
    npz_skipped_empty = sum(1 for item in writer_results if item.get("path") is None)
    npz_writes.extend(item for item in writer_results if item.get("path") is not None)
    raw_samples_added = sum(int(item["raw_rows"]) for item in npz_writes)
    samples_added = sum(int(item["effective_rows"]) for item in npz_writes)
    elapsed = perf_counter() - started
    games_done = completed_games + truncated_games
    if games_done > 0:
        # Game length in DECISIONS (incl. fast/policy-init plies), NOT recorded
        # training rows — the adaptive temperature half-life keys off real plies.
        latest_mean_length = total_decisions / games_done
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
        "total_decisions": total_decisions,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "search_visits": selfplay.search_visits,
        "pcr": {
            "enabled": pcr_enabled,
            "full_proportion": pcr_full_proportion if pcr_enabled else 1.0,
            "fast_visits": pcr_fast_visits if pcr_enabled else 0,
            "full_search_count": full_search_count,
            "fast_search_count": fast_search_count,
            "fast_rows_excluded": pcr_fast_rows_excluded,
            "npz_skipped_empty": npz_skipped_empty,
        },
        "policy_init": {
            "enabled": policy_init_enabled,
            "fraction": float(selfplay.policy_init_fraction),
            "avg_plies": float(selfplay.policy_init_avg_plies),
            "max_plies": int(selfplay.policy_init_max_plies),
            "temperature": float(selfplay.policy_init_temperature),
            "moves": policy_init_moves,
        },
        "root_policy_temperature_control": {
            "base": selfplay.root_policy_temperature,
            "early": root_temp_early,
            "halflife_plies": root_temp_halflife,
        },
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
    soft_z_lambda: float = 0.0,
    mask_opp_from_fast: bool = False,
) -> list[Model1SampleData]:
    return finalize_game_samples(
        pending,
        winner,
        horizons,
        truncated=truncated,
        soft_z_lambda=soft_z_lambda,
        mask_opp_from_fast=mask_opp_from_fast,
    )


def _player_label(value: object) -> str:
    return str(getattr(value, "value", value))
