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


@dataclass(frozen=True, slots=True)
class Model1ArchitectureConfig:
    input_channels: int = INPUT_CHANNELS
    channels: int = DEFAULT_CHANNELS
    residual_blocks: int = DEFAULT_BLOCKS
    dropout: float = 0.0
    short_term_value_horizons: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Model1TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
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
    mcts_session_cache_max_states: int = 1_048_576
    mcts_active_root_limit: int = 1024
    max_actions: int = 1024
    temperature: float = 1.0


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
    selfplay_batch_candidates: tuple[int, ...] = (1024,)
    training_batch_candidates: tuple[int, ...] = (64, 128, 192, 256)
    mcts_virtual_batch_candidates: tuple[int, ...] = (4,)
    selfplay_probe_positions: int = 8192
    probe_batches: int = 1


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
    return Model1Config(
        architecture=Model1ArchitectureConfig(
            input_channels=int(arch.get("input_channels", INPUT_CHANNELS)),
            channels=int(arch.get("channels", DEFAULT_CHANNELS)),
            residual_blocks=int(arch.get("residual_blocks", DEFAULT_BLOCKS)),
            dropout=float(arch.get("dropout", 0.0)),
            short_term_value_horizons=_int_tuple(arch.get("short_term_value_horizons", ())),
        ),
        training=Model1TrainingConfig(
            batch_size=int(training.get("batch_size", 128)),
            learning_rate=float(training.get("learning_rate", 1.0e-3)),
            weight_decay=float(training.get("weight_decay", 1.0e-4)),
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
            mcts_session_cache_max_states=int(selfplay.get("mcts_session_cache_max_states", 1_048_576)),
            mcts_active_root_limit=int(selfplay.get("mcts_active_root_limit", 1024)),
            max_actions=int(selfplay.get("max_actions", 1024)),
            temperature=float(selfplay.get("temperature", 1.0)),
        ),
        evaluation=Model1EvalConfig(
            games_per_epoch=int(evaluation.get("games_per_epoch", 64)),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=float(evaluation.get("sealbot_time_limit", 0.05)),
            max_actions=int(evaluation.get("max_actions", 1024)),
            require_sealbot=bool(evaluation.get("require_sealbot", False)),
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
