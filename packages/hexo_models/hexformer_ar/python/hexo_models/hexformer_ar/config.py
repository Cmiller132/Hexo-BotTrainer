"""Configuration for the Hexformer AR model family."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class HexformerArchitectureConfig:
    local_channels: int = 96
    local_blocks: int = 4
    token_dim: int = 256
    gps_layers: int = 6
    attention_heads: int = 8
    dropout: float = 0.05
    candidate_feature_dim: int = 24
    stone_feature_dim: int = 18
    window_feature_dim: int = 24
    global_feature_dim: int = 16
    local_input_channels: int = 13
    local_crop_size: int = 41
    max_local_windows: int = 3
    max_candidates: int = 768
    max_stones: int = 512
    max_windows: int = 768
    max_rel_edges: int = 4096
    rel_edge_feature_dim: int = 12
    lookahead_horizons: tuple[int, ...] = (1, 2, 4, 8)
    threat_classes: int = 6


@dataclass(frozen=True, slots=True)
class HexformerCandidateConfig:
    max_candidates: int = 768
    tactical_radius: int = 2
    recent_radius: int = 3
    frontier_radius: int = 8
    include_all_legal_below: int = 768
    require_tactical_candidates: bool = True


@dataclass(frozen=True, slots=True)
class HexformerTrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    policy_weight: float = 1.0
    wdl_weight: float = 1.0
    distance_weight: float = 0.25
    opponent_policy_weight: float = 0.15
    lookahead_weight: float = 0.25
    threat_weight: float = 0.5
    relevance_weight: float = 0.25
    symmetry_weight: float = 0.05
    amp: bool = True
    max_grad_norm: float = 1.0
    warmup_steps: int = 2_000
    cosine_decay_steps: int = 200_000
    min_learning_rate: float = 3.0e-5
    ema_decay: float = 0.999


@dataclass(frozen=True, slots=True)
class HexformerSamplesConfig:
    path: Path | None = None
    train_sample_count: int = 4096
    chunk_size: int = 4096
    recency_halflife: float = 100_000.0
    compression: str = "zlib"
    schema_version: int = 1
    hard_sample_fraction: float = 0.25
    recent_sample_fraction: float = 0.50


@dataclass(frozen=True, slots=True)
class HexformerSelfPlayConfig:
    samples_per_epoch: int = 4096
    games_per_epoch: int = 4096
    search_visits: int = 128
    max_actions: int = 1024
    c_puct: float = 1.5
    temperature: float = 1.0
    playout_cap_randomization: bool = True
    low_visit_probability: float = 0.25
    low_visit_count: int = 32


@dataclass(frozen=True, slots=True)
class HexformerEvalConfig:
    games_per_epoch: int = 64
    max_actions: int = 1024
    sealbot_variant: str = "best"
    sealbot_time_limit: float = 0.05
    require_sealbot: bool = False


@dataclass(frozen=True, slots=True)
class HexformerPerformanceConfig:
    calibrate: bool = True
    inference_batch_candidates: tuple[int, ...] = (16, 32, 64, 128)
    training_batch_candidates: tuple[int, ...] = (16, 32, 64)
    probe_batches: int = 1


@dataclass(frozen=True, slots=True)
class HexformerCurriculumConfig:
    synthetic_samples: int = 1024
    max_span: int = 8
    seed_offset: int = 910_337
    enabled_stages: tuple[str, ...] = ("win_in_1", "block_in_1", "double_threat")


@dataclass(frozen=True, slots=True)
class HexformerDebugConfig:
    write_sparse_batch_previews: bool = True
    write_candidate_diagnostics: bool = True
    write_tactical_diagnostics: bool = True
    preview_samples: int = 8


@dataclass(frozen=True, slots=True)
class HexformerConfig:
    architecture: HexformerArchitectureConfig = field(default_factory=HexformerArchitectureConfig)
    candidates: HexformerCandidateConfig = field(default_factory=HexformerCandidateConfig)
    training: HexformerTrainingConfig = field(default_factory=HexformerTrainingConfig)
    samples: HexformerSamplesConfig = field(default_factory=HexformerSamplesConfig)
    selfplay: HexformerSelfPlayConfig = field(default_factory=HexformerSelfPlayConfig)
    evaluation: HexformerEvalConfig = field(default_factory=HexformerEvalConfig)
    performance: HexformerPerformanceConfig = field(default_factory=HexformerPerformanceConfig)
    curriculum: HexformerCurriculumConfig = field(default_factory=HexformerCurriculumConfig)
    debug: HexformerDebugConfig = field(default_factory=HexformerDebugConfig)
    device: str = "cuda"
    checkpoint_path: Path | None = None


def parse_hexformer_config(raw: Mapping[str, Any] | None) -> HexformerConfig:
    config = dict(raw or {})
    arch = _section(config, "architecture")
    candidates = _section(config, "candidates")
    training = _section(config, "training")
    samples = _section(config, "samples")
    selfplay = _section(config, "selfplay")
    evaluation = _section(config, "evaluation")
    performance = _section(config, "performance")
    curriculum = _section(config, "curriculum")
    debug = _section(config, "debug")

    return HexformerConfig(
        architecture=HexformerArchitectureConfig(
            local_channels=int(arch.get("local_channels", 96)),
            local_blocks=int(arch.get("local_blocks", 4)),
            token_dim=int(arch.get("token_dim", 256)),
            gps_layers=int(arch.get("gps_layers", 6)),
            attention_heads=int(arch.get("attention_heads", 8)),
            dropout=float(arch.get("dropout", 0.05)),
            candidate_feature_dim=int(arch.get("candidate_feature_dim", 24)),
            stone_feature_dim=int(arch.get("stone_feature_dim", 18)),
            window_feature_dim=int(arch.get("window_feature_dim", 24)),
            global_feature_dim=int(arch.get("global_feature_dim", 16)),
            local_input_channels=int(arch.get("local_input_channels", 13)),
            local_crop_size=int(arch.get("local_crop_size", 41)),
            max_local_windows=int(arch.get("max_local_windows", 3)),
            max_candidates=int(arch.get("max_candidates", candidates.get("max_candidates", 768))),
            max_stones=int(arch.get("max_stones", 512)),
            max_windows=int(arch.get("max_windows", 768)),
            max_rel_edges=int(arch.get("max_rel_edges", 4096)),
            rel_edge_feature_dim=int(arch.get("rel_edge_feature_dim", 12)),
            lookahead_horizons=tuple(
                int(item) for item in arch.get("lookahead_horizons", (1, 2, 4, 8))
            ),
            threat_classes=int(arch.get("threat_classes", 6)),
        ),
        candidates=HexformerCandidateConfig(
            max_candidates=int(candidates.get("max_candidates", arch.get("max_candidates", 768))),
            tactical_radius=int(candidates.get("tactical_radius", 2)),
            recent_radius=int(candidates.get("recent_radius", 3)),
            frontier_radius=int(candidates.get("frontier_radius", 8)),
            include_all_legal_below=int(candidates.get("include_all_legal_below", 768)),
            require_tactical_candidates=bool(candidates.get("require_tactical_candidates", True)),
        ),
        training=HexformerTrainingConfig(
            batch_size=int(training.get("batch_size", 64)),
            learning_rate=float(training.get("learning_rate", 3.0e-4)),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
            policy_weight=float(training.get("policy_weight", 1.0)),
            wdl_weight=float(training.get("wdl_weight", 1.0)),
            distance_weight=float(training.get("distance_weight", 0.25)),
            opponent_policy_weight=float(training.get("opponent_policy_weight", 0.15)),
            lookahead_weight=float(training.get("lookahead_weight", 0.25)),
            threat_weight=float(training.get("threat_weight", 0.5)),
            relevance_weight=float(training.get("relevance_weight", 0.25)),
            symmetry_weight=float(training.get("symmetry_weight", 0.05)),
            amp=bool(training.get("amp", True)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            warmup_steps=int(training.get("warmup_steps", 2_000)),
            cosine_decay_steps=int(training.get("cosine_decay_steps", 200_000)),
            min_learning_rate=float(training.get("min_learning_rate", 3.0e-5)),
            ema_decay=float(training.get("ema_decay", 0.999)),
        ),
        samples=HexformerSamplesConfig(
            path=_optional_path(samples.get("path")),
            train_sample_count=int(samples.get("train_sample_count", 4096)),
            chunk_size=int(samples.get("chunk_size", 4096)),
            recency_halflife=float(samples.get("recency_halflife", 100_000.0)),
            compression=str(samples.get("compression", "zlib")),
            schema_version=int(samples.get("schema_version", 1)),
            hard_sample_fraction=float(samples.get("hard_sample_fraction", 0.25)),
            recent_sample_fraction=float(samples.get("recent_sample_fraction", 0.50)),
        ),
        selfplay=HexformerSelfPlayConfig(
            samples_per_epoch=int(selfplay.get("samples_per_epoch", 4096)),
            games_per_epoch=int(selfplay.get("games_per_epoch", 4096)),
            search_visits=int(selfplay.get("search_visits", 128)),
            max_actions=int(selfplay.get("max_actions", 1024)),
            c_puct=float(selfplay.get("c_puct", 1.5)),
            temperature=float(selfplay.get("temperature", 1.0)),
            playout_cap_randomization=bool(selfplay.get("playout_cap_randomization", True)),
            low_visit_probability=float(selfplay.get("low_visit_probability", 0.25)),
            low_visit_count=int(selfplay.get("low_visit_count", 32)),
        ),
        evaluation=HexformerEvalConfig(
            games_per_epoch=int(evaluation.get("games_per_epoch", 64)),
            max_actions=int(evaluation.get("max_actions", 1024)),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=float(evaluation.get("sealbot_time_limit", 0.05)),
            require_sealbot=bool(evaluation.get("require_sealbot", False)),
        ),
        performance=HexformerPerformanceConfig(
            calibrate=bool(performance.get("calibrate", True)),
            inference_batch_candidates=tuple(
                int(item) for item in performance.get("inference_batch_candidates", (16, 32, 64, 128))
            ),
            training_batch_candidates=tuple(
                int(item) for item in performance.get("training_batch_candidates", (16, 32, 64))
            ),
            probe_batches=int(performance.get("probe_batches", 1)),
        ),
        curriculum=HexformerCurriculumConfig(
            synthetic_samples=int(curriculum.get("synthetic_samples", 1024)),
            max_span=int(curriculum.get("max_span", 8)),
            seed_offset=int(curriculum.get("seed_offset", 910_337)),
            enabled_stages=tuple(
                str(item) for item in curriculum.get("enabled_stages", ("win_in_1", "block_in_1", "double_threat"))
            ),
        ),
        debug=HexformerDebugConfig(
            write_sparse_batch_previews=bool(debug.get("write_sparse_batch_previews", True)),
            write_candidate_diagnostics=bool(debug.get("write_candidate_diagnostics", True)),
            write_tactical_diagnostics=bool(debug.get("write_tactical_diagnostics", True)),
            preview_samples=int(debug.get("preview_samples", 8)),
        ),
        device=str(config.get("device", "cuda")),
        checkpoint_path=_optional_path(config.get("checkpoint_path")),
    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"hexformer config section {name!r} must be a mapping")
    return value


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))
