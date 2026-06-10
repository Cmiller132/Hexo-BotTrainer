"""Configuration objects for dense CNN Model 1.

`parse_model1_config` is the TOML boundary for this model. It rejects unknown
keys per section so a typo in a hand-authored config fails fast instead of being
silently ignored, then builds immutable dataclasses with light type coercion.

It deliberately does not re-validate every scalar's sign and range: the config
is authored by hand alongside the code, and the trainer, Rust session, and
optimizer raise clear errors for genuinely invalid values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import DEFAULT_BLOCKS, DEFAULT_CHANNELS, INPUT_CHANNELS


def _parse_temperature_schedule(raw: Any) -> tuple[tuple[int, float], ...]:
    """Coerce a TOML temperature schedule (list of [move, temperature] pairs).

    Returns a tuple of (move, temperature) sorted by move with unique, ascending
    moves and positive temperatures. Empty input yields an empty schedule (the
    linear ``final_temperature``/``temperature_decay_moves`` scheme stays in use).
    """

    if not raw:
        return ()
    points: list[tuple[int, float]] = []
    for item in raw:
        pair = tuple(item)
        if len(pair) != 2:
            raise ValueError("temperature_schedule entries must be [move, temperature] pairs")
        move = int(pair[0])
        temperature = float(pair[1])
        if move < 0:
            raise ValueError("temperature_schedule moves must be >= 0")
        if not temperature > 0.0:
            raise ValueError("temperature_schedule temperatures must be > 0")
        points.append((move, temperature))
    points.sort(key=lambda point: point[0])
    moves = [move for move, _ in points]
    if len(set(moves)) != len(moves):
        raise ValueError("temperature_schedule moves must have unique move indices")
    return tuple(points)


@dataclass(frozen=True, slots=True)
class Model1ArchitectureConfig:
    input_channels: int = INPUT_CHANNELS
    channels: int = DEFAULT_CHANNELS
    # ResTNet trunk interleave string, `_`-separated R/T blocks (see
    # architecture.parse_blocks_type). The 8-block R3-family default is
    # "R_R_R_T_R_R_T_R" (3 leading residual blocks, then a transformer every 3rd,
    # ending residual): RRR T RR T R. Override freely, e.g. the paper's canonical
    # 10-block R3(RRT) = "R_R_R_T_R_R_T_R_R_T".
    blocks_type: str = "R_R_R_T_R_R_T_R"
    attention_heads: int = 4
    mlp_ratio: int = 2
    # Stem conv kernel (EmbedNet): 3 (local mixing, default) or 1 (per-cell).
    embed_kernel_size: int = 3
    # Residual block conv: "hex" (hex-masked, board-correct, default) or "plain"
    # (literal nn.Conv2d paper match).
    residual_conv: str = "hex"
    # Attention implementation: "sdpa" (memory-efficient, identical math, default)
    # or "materialized" (the literal MSA_rel correctness oracle).
    attention_impl: str = "sdpa"
    # Attention scope: "full" (attend over all 1681 tokens, the original behavior),
    # "disk" (static -1e9 mask on the 420 out-of-crop corner KEY columns; default),
    # or "content" (gathered O(K^2) attention over occupied∪legal∩disk tokens).
    attention_scope: str = "disk"
    # Retained for back-compat / config symmetry with dense_cnn; the trunk depth
    # is governed by `blocks_type`, so this field is IGNORED by build_model.
    residual_blocks: int = DEFAULT_BLOCKS
    dropout: float = 0.0
    short_term_value_horizons: tuple[int, ...] = ()
    # Auxiliary moves-left head (predicts remaining decisions to game end, binned
    # over [0, MOVES_LEFT_CAP]). Training-only; shares the value reduction.
    moves_left_head: bool = True


@dataclass(frozen=True, slots=True)
class Model1TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    opp_policy_weight: float = 0.25
    short_term_value_weight: float = 0.25
    moves_left_weight: float = 0.1
    amp: bool = True
    max_grad_norm: float = 1.0
    train_samples_per_epoch: int = 100_000
    max_train_bucket_per_new_data: float = 8.0
    max_train_bucket_size: float = 500_000.0
    no_repeat_files: bool = True
    max_validation_samples: int = 100_000


@dataclass(frozen=True, slots=True)
class Model1SampleConfig:
    shuffle_min_rows: int = 100_000
    shuffle_keep_target_rows: int = 600_000
    shuffle_taper_window_exponent: float = 0.65
    shuffle_expand_window_per_row: float = 0.4
    shuffle_taper_window_scale: float = 50_000.0
    approx_rows_per_out_file: int = 70_000
    shuffle_worker_group_size: int = 80_000
    validation_fraction: float = 0.0
    policy_surprise_uniform_fraction: float = 0.5
    policy_surprise_max_weight: float = 8.0
    # Soft-Z value-target blend (Willemsen/Baier/Kaisers 2022):
    # `value = (1 - lambda) * z + lambda * root_value`, where z is the hard +1/-1
    # game outcome and root_value is the MCTS root value at the decision (already
    # side-to-move perspective). 0 (default) keeps the pure hard-outcome target;
    # a small lambda regularizes the value head against the bimodal +-1 labels of
    # an all-decisive game. See samples.finalize_game_samples.
    soft_z_lambda: float = 0.0


@dataclass(frozen=True, slots=True)
class Model1SelfPlayConfig:
    search_visits: int = 128
    active_games: int = 1024
    c_puct: float = 1.5
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_noise_fraction: float = 0.25
    root_dirichlet_total_alpha: float = 10.83
    root_policy_temperature: float = 1.1
    fpu_reduction: float = 0.20
    virtual_loss: float = 1.0
    widening_policy_mass: float = 0.95
    widening_max_children: int = 32
    widening_min_children: int = 2
    mcts_session_cache_max_states: int = 1_048_576
    mcts_active_root_limit: int = 1024
    scheduler: str = "lockstep"
    scheduler_flush_target: int = 0
    max_actions: int = 1024
    # Move-selection temperature SCHEDULE. `temperature` is the value used for the
    # opening; play linearly decays it to `final_temperature` over the first
    # `temperature_decay_moves` plies, then holds `final_temperature`. With
    # `temperature_decay_moves == 0` the schedule is flat at `temperature` (the old
    # behavior). AlphaZero/KataGo decay the opening temperature so endgames are
    # played decisively instead of sampled — keeping temperature high all game
    # produces noisy, weak-looking self-play tails.
    temperature: float = 1.0
    final_temperature: float = 1.0
    temperature_decay_moves: int = 0
    # Optional anchor-point temperature schedule. When non-empty it OVERRIDES the
    # linear `final_temperature`/`temperature_decay_moves` scheme: a tuple of
    # (move, temperature) points, linearly interpolated, holding the first point
    # before it and continuing the final segment's slope after the last point down
    # to `temperature_floor` (then held). Lets the opening stay hot while mid/late
    # play sharpens through chosen waypoints, e.g. [(0,1.0),(30,0.5),(90,0.3)].
    temperature_schedule: tuple[tuple[int, float], ...] = ()
    temperature_floor: float = 0.1
    # Adaptive exponential temperature decay (preferred over the anchor schemes
    # above; takes precedence when > 0). The played-move temperature is
    # `temperature * 0.5 ** (ply / (temperature_halflife_fraction * L))` clamped
    # at `temperature_floor`, where L is an EMA of the measured mean self-play
    # game length in decisions (persisted per run in selfplay/length_ema.json,
    # seeded with `temperature_length_prior` on a fresh run). The decay therefore
    # rescales itself as games lengthen/shorten — no absolute move anchors.
    temperature_halflife_fraction: float = 0.0
    temperature_length_prior: float = 32.0
    # Opening-diversity anchor (self-play). For the first `opening_moves` decisions
    # the played-move temperature is floored at `opening_temperature` (i.e.
    # `max(opening_temperature, base)`), holding a higher/flatter opening before the
    # adaptive decay takes over. The opening collapse is prior-driven and the
    # adaptive scheme already gives only ~0.8-1.0 early, so the anchor must be
    # meaningfully higher to diversify. 0 disables (default); never sharpens below
    # the base curve. Mirrors the eval-only opening_temperature/opening_moves.
    opening_temperature: float = 0.0
    opening_moves: int = 0
    # KataGo forced-playout strength (0 disables). Guarantees each materialized
    # root child ~sqrt(k * prior * root_visits) visits so Dirichlet root noise
    # survives PUCT into the visit counts (self-play opening diversity); the
    # exported policy target prunes the forced visits back out. k=2 is KataGo's.
    forced_playout_k: float = 0.0
    # KataGo root-policy temperature early ramp. When `root_policy_temperature_early`
    # > 0 the per-ply root softmax temperature interpolates exponentially from the
    # early value to the base `root_policy_temperature`:
    #   T(ply) = base + (early - base) * 0.5 ** (ply / halflife)
    # which is KataGo's interpolateEarly mechanism (its self-play config runs
    # rootPolicyTemperatureEarly=1.25 -> rootPolicyTemperature=1.1). The halflife
    # is in PLIES and must be > 0 when the ramp is enabled. 0 disables (constant
    # base temperature). Lockstep self-play only; eval/play uses the base scalar.
    root_policy_temperature_early: float = 0.0
    root_policy_temperature_halflife: float = 0.0
    # KataGo Playout Cap Randomization (Wu 2020). Each decision is independently a
    # FULL search with probability `pcr_full_proportion` (search_visits, recorded
    # as a training row, Dirichlet noise + forced playouts + temperature) or a
    # FAST search (`pcr_fast_visits`, NOT recorded, no noise, no forced playouts,
    # played greedily). Full positions get clean policy targets; fast moves buy
    # cheap game progression so more distinct games are generated per GPU-hour.
    # Mirrors the hexgt lineage implementation (HEXGT_PCR_KATAGO_MAPPING.md).
    # Lockstep scheduler only (fail-loud under continuous).
    pcr_enabled: bool = False
    pcr_full_proportion: float = 0.5
    pcr_fast_visits: int = 0
    # KataGo policy-initialized openings (initGamesWithPolicy). A fraction of
    # self-play games begin with a few opening plies SAMPLED DIRECTLY FROM THE RAW
    # NET PRIOR (no search): per selected game the ply count is drawn from a
    # truncated exponential with mean `policy_init_avg_plies`, capped at
    # `policy_init_max_plies`, and each such ply samples prior^(1/T) at
    # T=`policy_init_temperature` over all legal moves. Policy-init plies are NOT
    # recorded as training rows (there is no search target); they advance the game
    # so the value net continuously trains on off-distribution openings — the
    # KataGo lever against opening-monoculture self-reinforcement. The forced
    # origin ply (single legal move) is unaffected by construction. 0 disables.
    # Lockstep scheduler only (fail-loud under continuous).
    policy_init_fraction: float = 0.0
    policy_init_avg_plies: float = 0.0
    policy_init_max_plies: int = 0
    policy_init_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class Model1EvalConfig:
    games_per_epoch: int = 64
    sealbot_variant: str = "best"
    sealbot_time_limit: float = 0.05
    max_actions: int = 1024
    require_sealbot: bool = False
    # SealBot eval cadence: run the eval only on epochs where epoch % eval_every == 0.
    # 0 or 1 = every epoch (back-compat default). >1 amortizes the ~15-min SealBot eval
    # (e.g. 3 = eval every third epoch); skipped epochs write no eval diagnostic, so the
    # dashboard eval-trend simply has a gap.
    eval_every: int = 0
    # Opening diversification. With temperature 0 the dense player is fully
    # deterministic, so every eval game from the same start collapses to one
    # trajectory. Sampling the dense player's first `opening_moves` decisions at
    # `opening_temperature` (per-game seeded) spreads the openings so the win /
    # game-length signal reflects many distinct games. 0 keeps the old greedy
    # behavior.
    opening_temperature: float = 0.0
    opening_moves: int = 0
    # Eval/play-only MCTS virtual batch size. The single-game (player.decide)
    # search latency is bound by the NUMBER of NN forwards, not by selection:
    # raising the virtual batch fattens each forward and cuts latency ~linearly
    # in passes (measured ~3.4x from 4->32 at 512 sims) at a modest search-quality
    # cost (larger virtual-loss window). 0 keeps the calibrated self-play value.
    # This is the measured single-game latency lever — root parallelism and
    # shared-tree atomics were both benchmarked and found inferior.
    virtual_batch_size: int = 0


@dataclass(frozen=True, slots=True)
class Model1PerformanceConfig:
    calibrate: bool = True
    target_selfplay_positions_per_second: float = 128.0
    inference_batch_candidates: tuple[int, ...] = (128, 256, 512, 1024)
    selfplay_batch_candidates: tuple[int, ...] = (1024,)
    training_batch_candidates: tuple[int, ...] = (64, 128, 192, 256)
    mcts_virtual_batch_candidates: tuple[int, ...] = (4,)
    selfplay_probe_positions: int = 8192
    probe_batches: int = 1
    # Inference-forward optimizations (adopted, quality-safe). TensorRT FP16 is
    # correctness-gated at build time and falls back to the torch forward if the
    # gate fails or TRT is unavailable. Bucket pad multiple > 1 rounds the padded
    # forward batch up to that multiple instead of a power of two (less padding
    # waste); 0 keeps the power-of-two buckets. Both are equivalence-/quality-safe
    # (TRT gated; bucketing is pure batching). Env vars HEXO_TRT /
    # HEXO_BUCKET_PAD_MULTIPLE override these when set (launch-path escape hatch).
    inference_use_tensorrt: bool = False
    inference_fp16_model: bool = False
    inference_fp16_allow_fallback: bool = False
    inference_use_torch_compile: bool = False
    inference_compile_allow_torch_fallback: bool = False
    attention_kv_gather: bool = False
    inference_bucket_pad_multiple: int = 0
    # Fail-loud default: if TRT is on and the build/gate fails, abort with a
    # prominent error rather than silently running torch. Set true only to permit
    # a (still-logged) torch fallback. (env HEXO_TRT_ALLOW_FALLBACK overrides.)
    inference_trt_allow_torch_fallback: bool = False


@dataclass(frozen=True, slots=True)
class Model1Config:
    architecture: Model1ArchitectureConfig = field(default_factory=Model1ArchitectureConfig)
    training: Model1TrainingConfig = field(default_factory=Model1TrainingConfig)
    samples: Model1SampleConfig = field(default_factory=Model1SampleConfig)
    selfplay: Model1SelfPlayConfig = field(default_factory=Model1SelfPlayConfig)
    evaluation: Model1EvalConfig = field(default_factory=Model1EvalConfig)
    performance: Model1PerformanceConfig = field(default_factory=Model1PerformanceConfig)
    device: str = "cuda"
    checkpoint_path: Path | None = None


def parse_model1_config(raw: Mapping[str, Any] | None) -> Model1Config:
    """Parse the dense_cnn model section into immutable config dataclasses."""

    config = dict(raw or {})
    _reject_unknown(
        config,
        "model config",
        {"architecture", "training", "samples", "selfplay", "evaluation", "performance", "device", "checkpoint_path"},
    )
    arch = _section(config, "architecture", Model1ArchitectureConfig)
    training = _section(config, "training", Model1TrainingConfig)
    samples = _section(config, "samples", Model1SampleConfig)
    selfplay = _section(config, "selfplay", Model1SelfPlayConfig)
    evaluation = _section(config, "evaluation", Model1EvalConfig)
    performance = _section(config, "performance", Model1PerformanceConfig)

    checkpoint_path = config.get("checkpoint_path")
    use_trt = bool(performance.get("inference_use_tensorrt", False))
    use_compile = bool(performance.get("inference_use_torch_compile", False))
    if use_trt and use_compile:
        raise ValueError("inference_use_tensorrt and inference_use_torch_compile are mutually exclusive")
    return Model1Config(
        architecture=Model1ArchitectureConfig(
            input_channels=int(arch.get("input_channels", INPUT_CHANNELS)),
            channels=int(arch.get("channels", DEFAULT_CHANNELS)),
            blocks_type=str(arch.get("blocks_type", "R_R_R_T_R_R_T_R")),
            attention_heads=int(arch.get("attention_heads", 4)),
            mlp_ratio=int(arch.get("mlp_ratio", 2)),
            embed_kernel_size=int(arch.get("embed_kernel_size", 3)),
            residual_conv=str(arch.get("residual_conv", "hex")),
            attention_impl=str(arch.get("attention_impl", "sdpa")),
            attention_scope=str(arch.get("attention_scope", "disk")),
            residual_blocks=int(arch.get("residual_blocks", DEFAULT_BLOCKS)),
            dropout=float(arch.get("dropout", 0.0)),
            short_term_value_horizons=_even_horizons(arch.get("short_term_value_horizons", ())),
            moves_left_head=bool(arch.get("moves_left_head", True)),
        ),
        training=Model1TrainingConfig(
            batch_size=int(training.get("batch_size", 128)),
            learning_rate=float(training.get("learning_rate", 1.0e-3)),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
            policy_weight=float(training.get("policy_weight", 1.0)),
            value_weight=float(training.get("value_weight", 1.0)),
            opp_policy_weight=float(training.get("opp_policy_weight", 0.25)),
            short_term_value_weight=float(training.get("short_term_value_weight", 0.25)),
            moves_left_weight=float(training.get("moves_left_weight", 0.1)),
            amp=bool(training.get("amp", True)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            train_samples_per_epoch=int(training.get("train_samples_per_epoch", 100_000)),
            max_train_bucket_per_new_data=float(training.get("max_train_bucket_per_new_data", 8.0)),
            max_train_bucket_size=float(training.get("max_train_bucket_size", 500_000.0)),
            no_repeat_files=bool(training.get("no_repeat_files", True)),
            max_validation_samples=int(training.get("max_validation_samples", 100_000)),
        ),
        samples=Model1SampleConfig(
            shuffle_min_rows=int(samples.get("shuffle_min_rows", 100_000)),
            shuffle_keep_target_rows=int(samples.get("shuffle_keep_target_rows", 600_000)),
            shuffle_taper_window_exponent=float(samples.get("shuffle_taper_window_exponent", 0.65)),
            shuffle_expand_window_per_row=float(samples.get("shuffle_expand_window_per_row", 0.4)),
            shuffle_taper_window_scale=float(samples.get("shuffle_taper_window_scale", 50_000.0)),
            approx_rows_per_out_file=int(samples.get("approx_rows_per_out_file", 70_000)),
            shuffle_worker_group_size=int(samples.get("shuffle_worker_group_size", 80_000)),
            validation_fraction=float(samples.get("validation_fraction", 0.0)),
            policy_surprise_uniform_fraction=float(samples.get("policy_surprise_uniform_fraction", 0.5)),
            policy_surprise_max_weight=float(samples.get("policy_surprise_max_weight", 8.0)),
            soft_z_lambda=float(samples.get("soft_z_lambda", 0.0)),
        ),
        selfplay=Model1SelfPlayConfig(
            search_visits=int(selfplay.get("search_visits", 128)),
            active_games=int(selfplay.get("active_games", 1024)),
            c_puct=float(selfplay.get("c_puct", 1.5)),
            root_dirichlet_noise_enabled=bool(selfplay.get("root_dirichlet_noise_enabled", True)),
            root_dirichlet_noise_fraction=float(selfplay.get("root_dirichlet_noise_fraction", 0.25)),
            root_dirichlet_total_alpha=float(selfplay.get("root_dirichlet_total_alpha", 10.83)),
            root_policy_temperature=float(selfplay.get("root_policy_temperature", 1.1)),
            fpu_reduction=float(selfplay.get("fpu_reduction", 0.20)),
            virtual_loss=float(selfplay.get("virtual_loss", 1.0)),
            widening_policy_mass=float(selfplay.get("widening_policy_mass", 0.95)),
            widening_max_children=int(selfplay.get("widening_max_children", 32)),
            widening_min_children=int(selfplay.get("widening_min_children", 2)),
            mcts_session_cache_max_states=int(selfplay.get("mcts_session_cache_max_states", 1_048_576)),
            mcts_active_root_limit=int(selfplay.get("mcts_active_root_limit", 1024)),
            scheduler=_scheduler_name(selfplay.get("scheduler", "lockstep")),
            scheduler_flush_target=_scheduler_flush_target(selfplay.get("scheduler_flush_target", 0)),
            max_actions=int(selfplay.get("max_actions", 1024)),
            temperature=float(selfplay.get("temperature", 1.0)),
            final_temperature=float(selfplay.get("final_temperature", selfplay.get("temperature", 1.0))),
            temperature_decay_moves=int(selfplay.get("temperature_decay_moves", 0)),
            temperature_schedule=_parse_temperature_schedule(selfplay.get("temperature_schedule", ())),
            temperature_floor=float(selfplay.get("temperature_floor", 0.1)),
            temperature_halflife_fraction=float(selfplay.get("temperature_halflife_fraction", 0.0)),
            temperature_length_prior=float(selfplay.get("temperature_length_prior", 32.0)),
            opening_temperature=float(selfplay.get("opening_temperature", 0.0)),
            opening_moves=int(selfplay.get("opening_moves", 0)),
            forced_playout_k=float(selfplay.get("forced_playout_k", 0.0)),
            root_policy_temperature_early=float(selfplay.get("root_policy_temperature_early", 0.0)),
            root_policy_temperature_halflife=float(selfplay.get("root_policy_temperature_halflife", 0.0)),
            pcr_enabled=bool(selfplay.get("pcr_enabled", False)),
            pcr_full_proportion=float(selfplay.get("pcr_full_proportion", 0.5)),
            pcr_fast_visits=int(selfplay.get("pcr_fast_visits", 0)),
            policy_init_fraction=float(selfplay.get("policy_init_fraction", 0.0)),
            policy_init_avg_plies=float(selfplay.get("policy_init_avg_plies", 0.0)),
            policy_init_max_plies=int(selfplay.get("policy_init_max_plies", 0)),
            policy_init_temperature=float(selfplay.get("policy_init_temperature", 1.0)),
        ),
        evaluation=Model1EvalConfig(
            games_per_epoch=int(evaluation.get("games_per_epoch", 64)),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=float(evaluation.get("sealbot_time_limit", 0.05)),
            max_actions=int(evaluation.get("max_actions", 1024)),
            require_sealbot=bool(evaluation.get("require_sealbot", False)),
            eval_every=int(evaluation.get("eval_every", 0)),
            opening_temperature=float(evaluation.get("opening_temperature", 0.0)),
            opening_moves=int(evaluation.get("opening_moves", 0)),
            virtual_batch_size=int(evaluation.get("virtual_batch_size", 0)),
        ),
        performance=Model1PerformanceConfig(
            calibrate=bool(performance.get("calibrate", True)),
            target_selfplay_positions_per_second=float(performance.get("target_selfplay_positions_per_second", 128.0)),
            inference_batch_candidates=_int_tuple(performance.get("inference_batch_candidates", (128, 256, 512, 1024))),
            selfplay_batch_candidates=_int_tuple(performance.get("selfplay_batch_candidates", (1024,))),
            training_batch_candidates=_int_tuple(performance.get("training_batch_candidates", (64, 128, 192, 256))),
            mcts_virtual_batch_candidates=_int_tuple(performance.get("mcts_virtual_batch_candidates", (4,))),
            selfplay_probe_positions=int(performance.get("selfplay_probe_positions", 8192)),
            probe_batches=int(performance.get("probe_batches", 1)),
            inference_use_tensorrt=use_trt,
            inference_fp16_model=bool(performance.get("inference_fp16_model", False)),
            inference_fp16_allow_fallback=bool(performance.get("inference_fp16_allow_fallback", False)),
            inference_use_torch_compile=use_compile,
            inference_compile_allow_torch_fallback=bool(performance.get("inference_compile_allow_torch_fallback", False)),
            attention_kv_gather=bool(performance.get("attention_kv_gather", False)),
            inference_bucket_pad_multiple=int(performance.get("inference_bucket_pad_multiple", 0)),
            inference_trt_allow_torch_fallback=bool(performance.get("inference_trt_allow_torch_fallback", False)),
        ),
        device=str(config.get("device", "cuda")),
        checkpoint_path=Path(str(checkpoint_path)) if checkpoint_path else None,
    )


def _section(raw: Mapping[str, Any], name: str, dataclass_type: type) -> Mapping[str, Any]:
    """Return a config subsection, rejecting keys the dataclass does not define."""

    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"model config section {name!r} must be a mapping")
    _reject_unknown(value, f"model config section {name!r}", set(dataclass_type.__dataclass_fields__))
    return value


def _reject_unknown(raw: Mapping[str, Any], label: str, allowed: set[str]) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported key(s): {', '.join(unknown)}")


def _int_tuple(value: Sequence[int] | Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"expected a sequence of integers, got {value!r}")
    return tuple(int(item) for item in value)


def _even_horizons(value: Sequence[int] | Any) -> tuple[int, ...]:
    """Lookahead horizons must be positive EVEN decision counts (full turns).

    Every turn after the opening is two decisions by the same player; the
    short-term-value EMA steps over full turns (see
    ``samples._short_term_value_targets``), so an odd horizon has no meaning.
    """

    horizons = _int_tuple(value)
    bad = [item for item in horizons if item <= 0 or item % 2 != 0]
    if bad:
        raise ValueError(
            f"short_term_value_horizons must be positive even decision counts (full turns), got {bad!r}"
        )
    return horizons


def _scheduler_name(value: Any) -> str:
    scheduler = str(value)
    if scheduler not in ("lockstep", "continuous"):
        raise ValueError("selfplay.scheduler must be 'lockstep' or 'continuous'")
    return scheduler


def _scheduler_flush_target(value: Any) -> int:
    target = int(value)
    if target < 0:
        raise ValueError("selfplay.scheduler_flush_target must be >= 0 (0 = calibrated inference batch)")
    return target
