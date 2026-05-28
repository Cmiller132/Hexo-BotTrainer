"""Configuration objects for the Model 1 production path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .constants import DEFAULT_BLOCKS, DEFAULT_CHANNELS, INPUT_CHANNELS, BOARD_SIZE


@dataclass(frozen=True, slots=True)
class Model1ArchitectureConfig:
    input_channels: int = INPUT_CHANNELS
    channels: int = DEFAULT_CHANNELS
    residual_blocks: int = DEFAULT_BLOCKS
    crop_size: int = BOARD_SIZE
    dropout: float = 0.0
    lookahead_horizons: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Model1TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    opp_policy_weight: float = 0.25
    lookahead_weight: float = 0.25
    amp: bool = True
    max_grad_norm: float = 1.0


@dataclass(frozen=True, slots=True)
class Model1SampleConfig:
    capacity: int = 200_000
    train_sample_count: int = 4096
    recency_halflife: float = 50_000.0
    compression_level: int = 6


@dataclass(frozen=True, slots=True)
class Model1SelfPlayConfig:
    samples_per_epoch: int = 4096
    search_visits: int = 128
    active_games: int = 2048
    progressive_widening_initial_actions: int = 128
    progressive_widening_child_initial_actions: int = 32
    progressive_widening_candidate_actions: int = 192
    progressive_widening_growth_interval: float = 40.0
    progressive_widening_growth_base: float = 1.3
    mcts_evaluation_cache_max_states: int = 131_072
    mcts_active_root_limit: int = 512
    max_actions: int = 1024
    temperature: float = 1.0
    worker_count: int = 1


@dataclass(frozen=True, slots=True)
class Model1EvalConfig:
    games_per_epoch: int = 64
    sealbot_variant: str = "best"
    sealbot_time_limit: float = 0.05
    max_actions: int = 1024
    require_sealbot: bool = False


@dataclass(frozen=True, slots=True)
class Model1PerformanceConfig:
    calibrate: bool = True
    target_selfplay_positions_per_second: float = 128.0
    inference_batch_candidates: tuple[int, ...] = (128, 256, 512, 1024)
    selfplay_batch_candidates: tuple[int, ...] = (2048,)
    training_batch_candidates: tuple[int, ...] = (64, 128, 192, 256)
    mcts_visit_candidates: tuple[int, ...] = (128,)
    mcts_virtual_batch_candidates: tuple[int, ...] = (4,)
    selfplay_probe_positions: int = 4096
    probe_batches: int = 1


@dataclass(frozen=True, slots=True)
class Model1DebugConfig:
    write_game_history: bool = True
    write_policy_targets: bool = True
    write_sample_previews: bool = True
    preview_games: int = 4


@dataclass(frozen=True, slots=True)
class Model1Config:
    architecture: Model1ArchitectureConfig = field(default_factory=Model1ArchitectureConfig)
    training: Model1TrainingConfig = field(default_factory=Model1TrainingConfig)
    samples: Model1SampleConfig = field(default_factory=Model1SampleConfig)
    selfplay: Model1SelfPlayConfig = field(default_factory=Model1SelfPlayConfig)
    evaluation: Model1EvalConfig = field(default_factory=Model1EvalConfig)
    performance: Model1PerformanceConfig = field(default_factory=Model1PerformanceConfig)
    debug: Model1DebugConfig = field(default_factory=Model1DebugConfig)
    device: str = "cuda"
    checkpoint_path: Path | None = None


def parse_model1_config(raw: Mapping[str, Any] | None) -> Model1Config:
    config = dict(raw or {})
    arch = _section(config, "architecture")
    training = _section(config, "training")
    samples = _section(config, "samples")
    selfplay = _section(config, "selfplay")
    evaluation = _section(config, "evaluation")
    performance = _section(config, "performance")
    debug = _section(config, "debug")

    return Model1Config(
        architecture=Model1ArchitectureConfig(
            input_channels=int(arch.get("input_channels", config.get("input_channels", INPUT_CHANNELS))),
            channels=int(arch.get("channels", config.get("channels", DEFAULT_CHANNELS))),
            residual_blocks=int(arch.get("residual_blocks", arch.get("blocks", config.get("blocks", DEFAULT_BLOCKS)))),
            crop_size=int(arch.get("crop_size", config.get("crop_size", BOARD_SIZE))),
            dropout=float(arch.get("dropout", 0.0)),
            lookahead_horizons=tuple(int(item) for item in arch.get("lookahead_horizons", config.get("lookahead_horizons", ()))),
        ),
        training=Model1TrainingConfig(
            batch_size=int(training.get("batch_size", 128)),
            learning_rate=float(training.get("learning_rate", 1.0e-3)),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
            policy_weight=float(training.get("policy_weight", 1.0)),
            value_weight=float(training.get("value_weight", 1.0)),
            opp_policy_weight=float(training.get("opp_policy_weight", 0.25)),
            lookahead_weight=float(training.get("lookahead_weight", 0.25)),
            amp=bool(training.get("amp", True)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        ),
        samples=Model1SampleConfig(
            capacity=max(200_000, int(samples.get("capacity", 200_000))),
            train_sample_count=int(samples.get("train_sample_count", config.get("train_sample_count", 4096))),
            recency_halflife=float(samples.get("recency_halflife", 50_000.0)),
            compression_level=int(samples.get("compression_level", 6)),
        ),
        selfplay=Model1SelfPlayConfig(
            samples_per_epoch=int(selfplay.get("samples_per_epoch", 4096)),
            search_visits=int(selfplay.get("search_visits", 128)),
            active_games=int(selfplay.get("active_games", selfplay.get("batch_size", 2048))),
            progressive_widening_initial_actions=max(
                1,
                int(selfplay.get("progressive_widening_initial_actions", 128)),
            ),
            progressive_widening_child_initial_actions=max(
                1,
                int(selfplay.get("progressive_widening_child_initial_actions", 32)),
            ),
            progressive_widening_candidate_actions=max(
                1,
                int(selfplay.get("progressive_widening_candidate_actions", 192)),
            ),
            progressive_widening_growth_interval=max(
                1.0,
                float(selfplay.get("progressive_widening_growth_interval", 40.0)),
            ),
            progressive_widening_growth_base=max(
                1.000001,
                float(selfplay.get("progressive_widening_growth_base", 1.3)),
            ),
            mcts_evaluation_cache_max_states=max(
                1,
                int(selfplay.get("mcts_evaluation_cache_max_states", 131_072)),
            ),
            mcts_active_root_limit=max(1, int(selfplay.get("mcts_active_root_limit", 512))),
            max_actions=int(selfplay.get("max_actions", 1024)),
            temperature=float(selfplay.get("temperature", 1.0)),
            worker_count=int(selfplay.get("worker_count", 1)),
        ),
        evaluation=Model1EvalConfig(
            games_per_epoch=int(evaluation.get("games_per_epoch", 64)),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=float(evaluation.get("sealbot_time_limit", 0.05)),
            max_actions=int(evaluation.get("max_actions", 1024)),
            require_sealbot=bool(evaluation.get("require_sealbot", False)),
        ),
        performance=Model1PerformanceConfig(
            calibrate=bool(performance.get("calibrate", performance.get("calibration_enabled", True))),
            target_selfplay_positions_per_second=float(performance.get("target_selfplay_positions_per_second", 128.0)),
            inference_batch_candidates=tuple(int(item) for item in performance.get("inference_batch_candidates", (128, 256, 512, 1024))),
            selfplay_batch_candidates=tuple(int(item) for item in performance.get("selfplay_batch_candidates", (2048,))),
            training_batch_candidates=tuple(int(item) for item in performance.get("training_batch_candidates", (64, 128, 192, 256))),
            mcts_visit_candidates=tuple(int(item) for item in performance.get("mcts_visit_candidates", (selfplay.get("search_visits", 128),))),
            mcts_virtual_batch_candidates=tuple(int(item) for item in performance.get("mcts_virtual_batch_candidates", (4,))),
            selfplay_probe_positions=max(1, int(performance.get("selfplay_probe_positions", 4096))),
            probe_batches=max(1, int(performance.get("probe_batches", performance.get("calibration_measurement_batches", 1)))),
        ),
        debug=Model1DebugConfig(
            write_game_history=bool(debug.get("write_game_history", True)),
            write_policy_targets=bool(debug.get("write_policy_targets", True)),
            write_sample_previews=bool(debug.get("write_sample_previews", True)),
            preview_games=int(debug.get("preview_games", 4)),
        ),
        device=str(config.get("device", "cuda")),
        checkpoint_path=_optional_path(config.get("checkpoint_path")),
    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"model config section {name!r} must be a mapping")
    return value


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))
