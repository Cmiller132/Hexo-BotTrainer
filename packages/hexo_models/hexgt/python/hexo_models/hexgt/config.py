"""Configuration objects for hexgt (Model 2).

`parse_hexgt_config` is the TOML boundary for this model. It mirrors dense_cnn's
discipline: reject unknown keys per section so a typo fails fast, then build
immutable dataclasses with light type coercion and no per-scalar range checks.

The pipeline knobs (training / samples / selfplay / evaluation / performance)
stay close to dense_cnn because Model 2 slots into the same training / MCTS /
replay / eval pipeline (drop-in PIPELINE compatibility, §2). Only the
`architecture` section differs: it describes a dynamic GNN + transformer instead
of a dense CNN, and carries `candidate_radius` (the single `n`, §4) which is
threaded into BOTH sample-gen and live MCTS so the move vocabulary matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import (
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_CANDIDATE_RADIUS,
    DEFAULT_CTX_LAYERS,
    DEFAULT_FFN_DIM,
    DEFAULT_GNN_LAYERS,
    DEFAULT_TOKEN_DIM,
    DEFAULT_VALUE_PMA_SEEDS,
    NODE_FEATURE_DIM,
)


def _parse_temperature_schedule(raw: Any) -> tuple[tuple[int, float], ...]:
    """Coerce a TOML temperature schedule (list of [move, temperature] pairs)."""

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
class HexgtArchitectureConfig:
    """Dynamic GNN + transformer hyperparameters and the candidate rule `n`."""

    node_feature_dim: int = NODE_FEATURE_DIM
    token_dim: int = DEFAULT_TOKEN_DIM
    gnn_layers: int = DEFAULT_GNN_LAYERS
    ctx_layers: int = DEFAULT_CTX_LAYERS
    attention_heads: int = DEFAULT_ATTENTION_HEADS
    ffn_dim: int = DEFAULT_FFN_DIM
    dropout: float = 0.0
    short_term_value_horizons: tuple[int, ...] = ()
    # The single candidate-set neighborhood radius `n` (§4). Threaded into both
    # sample-gen and live MCTS so training support == search expansion.
    candidate_radius: int = DEFAULT_CANDIDATE_RADIUS
    # PMA value-head seed count k (HEXGT_PMA_VALUE_HEAD_PLAN.md). The value readout
    # is [SIDE | PMA_k]; k=2 is the owner's chosen build (doc default is 1).
    value_pma_seeds: int = DEFAULT_VALUE_PMA_SEEDS
    # Whether the value head concatenates the SIDE-hub embedding before the PMA pool
    # (`[SIDE | PMA_k]`, width (1+k)*token_dim). False -> PMA-only (`[PMA_k]`, width
    # k*token_dim), an A/B toggle to isolate the PMA pool's contribution. Default
    # True keeps the current build; a no-SIDE run retrains from a pretrain seed.
    value_head_use_side: bool = True


@dataclass(frozen=True, slots=True)
class HexgtTrainingConfig:
    batch_size: int = 128
    # GNN/transformer wants a lower LR + warmup than a CNN (readiness gap F).
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000
    policy_weight: float = 1.0
    value_weight: float = 1.0
    opp_policy_weight: float = 0.25
    short_term_value_weight: float = 0.25
    amp: bool = True
    max_grad_norm: float = 1.0
    train_samples_per_epoch: int = 100_000
    max_train_bucket_per_new_data: float = 8.0
    max_train_bucket_size: float = 500_000.0
    no_repeat_files: bool = True
    max_validation_samples: int = 100_000


@dataclass(frozen=True, slots=True)
class HexgtSampleConfig:
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
    # KataGo policy-surprise sample weighting via row duplication (KL(visits||prior)).
    # When enabled, self-play repeats each finalized position by a frequency weight
    # mixing a uniform floor (`policy_surprise_uniform_fraction`) with a term
    # proportional to its policy surprise, clamped to `policy_surprise_max_weight`, so
    # positions where the search most disagreed with the network prior carry more
    # training signal. OFF by default so the existing/halted lineages are unchanged;
    # the active run opts in via its TOML. Interacts cleanly with the recency-weighted
    # shard sampling (per-shard) and STV/soft-Z targets (per-row, copied verbatim).
    policy_surprise_enabled: bool = False
    # Dataset pruning for BC/sample-gen (candidate_radius decision): a recorded
    # position is PRUNED when the fraction of its policy visit-mass that lands
    # OUTSIDE the n-radius candidate set exceeds this threshold (a far-spread move
    # the new rep deliberately cannot represent). Surviving positions renormalize
    # over the in-set candidates. Set to 1.0 to disable pruning.
    bc_prune_max_dropped_mass: float = 0.15
    # Soft-Z value target (Willemsen/Baier/Kaisers 2022): the main value label is
    # the convex blend ``(1 - soft_z_lambda) * z + soft_z_lambda * root_value``
    # where ``z`` is the hard game outcome and ``root_value`` is the MCTS root
    # value at the position. Recalibrates the saturated +-1 label / +0.82 optimism
    # bias. 0.0 == pure hard outcome; start 0.5, anneal ~0.3->0.7 (see
    # docs/analysis/HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md PART 2).
    soft_z_lambda: float = 0.5


@dataclass(frozen=True, slots=True)
class HexgtSelfPlayConfig:
    search_visits: int = 128
    active_games: int = 1024
    c_puct: float = 1.5
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_noise_fraction: float = 0.25
    root_dirichlet_total_alpha: float = 10.83
    root_policy_temperature: float = 1.1
    fpu_reduction: float = 0.20
    virtual_loss: float = 1.0
    # MCTS nucleus (top-p) widening within the model's full candidate support
    # (§6.6). Identical between self-play and eval for a fair comparison.
    widening_policy_mass: float = 0.95
    widening_max_children: int = 32
    widening_min_children: int = 2
    mcts_session_cache_max_states: int = 1_048_576
    mcts_active_root_limit: int = 1024
    max_actions: int = 1024
    temperature: float = 1.0
    final_temperature: float = 1.0
    temperature_decay_moves: int = 0
    temperature_schedule: tuple[tuple[int, float], ...] = ()
    temperature_floor: float = 0.1
    # KataGo-style smooth exponential-halflife move temperature. When > 0 it takes
    # precedence over the linear/`temperature_schedule` decay:
    #   temp(ply) = temperature_floor + (temperature - temperature_floor) * 2**(-ply/halflife)
    # i.e. starts at `temperature`, decays by half every `temperature_halflife` plies,
    # and asymptotes to `temperature_floor` (the honored late-game floor).
    temperature_halflife: float = 0.0
    forced_playout_k: float = 0.0


@dataclass(frozen=True, slots=True)
class HexgtEvalConfig:
    games_per_epoch: int = 64
    sealbot_variant: str = "best"
    sealbot_time_limit: float = 0.05
    max_actions: int = 1024
    require_sealbot: bool = False
    opening_temperature: float = 0.0
    opening_moves: int = 0
    virtual_batch_size: int = 0


@dataclass(frozen=True, slots=True)
class HexgtPerformanceConfig:
    calibrate: bool = True
    target_selfplay_positions_per_second: float = 128.0
    inference_batch_candidates: tuple[int, ...] = (128, 256, 512, 1024)
    selfplay_batch_candidates: tuple[int, ...] = (1024,)
    training_batch_candidates: tuple[int, ...] = (64, 128, 192, 256)
    mcts_virtual_batch_candidates: tuple[int, ...] = (4,)
    selfplay_probe_positions: int = 8192
    probe_batches: int = 1
    # No TensorRT for a truly dynamic GNN (§6.1) — torch FP16 only. The TRT knobs
    # are intentionally absent from this model's config.


@dataclass(frozen=True, slots=True)
class HexgtConfig:
    architecture: HexgtArchitectureConfig = field(default_factory=HexgtArchitectureConfig)
    training: HexgtTrainingConfig = field(default_factory=HexgtTrainingConfig)
    samples: HexgtSampleConfig = field(default_factory=HexgtSampleConfig)
    selfplay: HexgtSelfPlayConfig = field(default_factory=HexgtSelfPlayConfig)
    evaluation: HexgtEvalConfig = field(default_factory=HexgtEvalConfig)
    performance: HexgtPerformanceConfig = field(default_factory=HexgtPerformanceConfig)
    device: str = "cuda"
    checkpoint_path: Path | None = None


def parse_hexgt_config(raw: Mapping[str, Any] | None) -> HexgtConfig:
    """Parse the hexgt model section into immutable config dataclasses."""

    config = dict(raw or {})
    _reject_unknown(
        config,
        "model config",
        {"architecture", "training", "samples", "selfplay", "evaluation", "performance", "device", "checkpoint_path"},
    )
    arch = _section(config, "architecture", HexgtArchitectureConfig)
    training = _section(config, "training", HexgtTrainingConfig)
    samples = _section(config, "samples", HexgtSampleConfig)
    selfplay = _section(config, "selfplay", HexgtSelfPlayConfig)
    evaluation = _section(config, "evaluation", HexgtEvalConfig)
    performance = _section(config, "performance", HexgtPerformanceConfig)

    checkpoint_path = config.get("checkpoint_path")
    return HexgtConfig(
        architecture=HexgtArchitectureConfig(
            node_feature_dim=int(arch.get("node_feature_dim", NODE_FEATURE_DIM)),
            token_dim=int(arch.get("token_dim", DEFAULT_TOKEN_DIM)),
            gnn_layers=int(arch.get("gnn_layers", DEFAULT_GNN_LAYERS)),
            ctx_layers=int(arch.get("ctx_layers", DEFAULT_CTX_LAYERS)),
            attention_heads=int(arch.get("attention_heads", DEFAULT_ATTENTION_HEADS)),
            ffn_dim=int(arch.get("ffn_dim", DEFAULT_FFN_DIM)),
            dropout=float(arch.get("dropout", 0.0)),
            short_term_value_horizons=_int_tuple(arch.get("short_term_value_horizons", ())),
            candidate_radius=int(arch.get("candidate_radius", DEFAULT_CANDIDATE_RADIUS)),
            value_pma_seeds=int(arch.get("value_pma_seeds", DEFAULT_VALUE_PMA_SEEDS)),
            value_head_use_side=bool(arch.get("value_head_use_side", True)),
        ),
        training=HexgtTrainingConfig(
            batch_size=int(training.get("batch_size", 128)),
            learning_rate=float(training.get("learning_rate", 3.0e-4)),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
            warmup_steps=int(training.get("warmup_steps", 1000)),
            policy_weight=float(training.get("policy_weight", 1.0)),
            value_weight=float(training.get("value_weight", 1.0)),
            opp_policy_weight=float(training.get("opp_policy_weight", 0.25)),
            short_term_value_weight=float(training.get("short_term_value_weight", 0.25)),
            amp=bool(training.get("amp", True)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            train_samples_per_epoch=int(training.get("train_samples_per_epoch", 100_000)),
            max_train_bucket_per_new_data=float(training.get("max_train_bucket_per_new_data", 8.0)),
            max_train_bucket_size=float(training.get("max_train_bucket_size", 500_000.0)),
            no_repeat_files=bool(training.get("no_repeat_files", True)),
            max_validation_samples=int(training.get("max_validation_samples", 100_000)),
        ),
        samples=HexgtSampleConfig(
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
            policy_surprise_enabled=bool(samples.get("policy_surprise_enabled", False)),
            bc_prune_max_dropped_mass=float(samples.get("bc_prune_max_dropped_mass", 0.15)),
            soft_z_lambda=float(samples.get("soft_z_lambda", 0.5)),
        ),
        selfplay=HexgtSelfPlayConfig(
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
            max_actions=int(selfplay.get("max_actions", 1024)),
            temperature=float(selfplay.get("temperature", 1.0)),
            final_temperature=float(selfplay.get("final_temperature", selfplay.get("temperature", 1.0))),
            temperature_decay_moves=int(selfplay.get("temperature_decay_moves", 0)),
            temperature_schedule=_parse_temperature_schedule(selfplay.get("temperature_schedule", ())),
            temperature_floor=float(selfplay.get("temperature_floor", 0.1)),
            temperature_halflife=float(selfplay.get("temperature_halflife", 0.0)),
            forced_playout_k=float(selfplay.get("forced_playout_k", 0.0)),
        ),
        evaluation=HexgtEvalConfig(
            games_per_epoch=int(evaluation.get("games_per_epoch", 64)),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=float(evaluation.get("sealbot_time_limit", 0.05)),
            max_actions=int(evaluation.get("max_actions", 1024)),
            require_sealbot=bool(evaluation.get("require_sealbot", False)),
            opening_temperature=float(evaluation.get("opening_temperature", 0.0)),
            opening_moves=int(evaluation.get("opening_moves", 0)),
            virtual_batch_size=int(evaluation.get("virtual_batch_size", 0)),
        ),
        performance=HexgtPerformanceConfig(
            calibrate=bool(performance.get("calibrate", True)),
            target_selfplay_positions_per_second=float(performance.get("target_selfplay_positions_per_second", 128.0)),
            inference_batch_candidates=_int_tuple(performance.get("inference_batch_candidates", (128, 256, 512, 1024))),
            selfplay_batch_candidates=_int_tuple(performance.get("selfplay_batch_candidates", (1024,))),
            training_batch_candidates=_int_tuple(performance.get("training_batch_candidates", (64, 128, 192, 256))),
            mcts_virtual_batch_candidates=_int_tuple(performance.get("mcts_virtual_batch_candidates", (4,))),
            selfplay_probe_positions=int(performance.get("selfplay_probe_positions", 8192)),
            probe_batches=int(performance.get("probe_batches", 1)),
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
