"""Strict configuration objects for dense CNN Model 1.

`parse_model1_config` is the TOML boundary for this model. It accepts only the
current production sections and keys, converts scalars into typed dataclasses,
and raises `ValueError` for unknown, non-finite, out-of-range, or incompatible
values.

The small validation helpers in this file are intentionally centralized here:
configuration is a real external boundary, and raising clear config errors here
prevents invalid settings from reaching the trainer, Rust MCTS session, or
PyTorch optimizer with less useful failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

from .constants import DEFAULT_BLOCKS, DEFAULT_CHANNELS, INPUT_CHANNELS


@dataclass(frozen=True, slots=True)
class Model1ArchitectureConfig:
    input_channels: int = INPUT_CHANNELS
    channels: int = DEFAULT_CHANNELS
    residual_blocks: int = DEFAULT_BLOCKS
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
    min_mcts_samples_per_game: int = 32
    progressive_widening_initial_actions: int = 8
    progressive_widening_child_initial_actions: int = 4
    progressive_widening_candidate_actions: int = 128
    progressive_widening_growth_interval: float = 256.0
    progressive_widening_growth_base: float = 1.3
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_noise_fraction: float = 0.25
    root_dirichlet_alpha: float = 0.03
    hidden_prior_mass: float = 0.05
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
    selfplay_batch_candidates: tuple[int, ...] = (2048,)
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
    """Parse the dense_cnn model section into immutable config dataclasses.

    This function is intentionally explicit rather than reflection-driven. Each
    allowed key is listed at the parse site, which makes deleted production
    options fail as unknown keys instead of lingering as decorative config.
    """

    config = dict(raw or {})
    _reject_unknown(
        config,
        "model config",
        {
            "architecture",
            "training",
            "samples",
            "selfplay",
            "evaluation",
            "performance",
            "device",
            "checkpoint_path",
        },
    )
    arch = _section(
        config,
        "architecture",
        {"input_channels", "channels", "residual_blocks", "dropout", "lookahead_horizons"},
    )
    training = _section(
        config,
        "training",
        {
            "batch_size",
            "learning_rate",
            "weight_decay",
            "policy_weight",
            "value_weight",
            "opp_policy_weight",
            "lookahead_weight",
            "amp",
            "max_grad_norm",
        },
    )
    samples = _section(
        config,
        "samples",
        {"capacity", "train_sample_count", "recency_halflife", "compression_level"},
    )
    selfplay = _section(
        config,
        "selfplay",
        {
            "samples_per_epoch",
            "search_visits",
            "active_games",
            "min_mcts_samples_per_game",
            "progressive_widening_initial_actions",
            "progressive_widening_child_initial_actions",
            "progressive_widening_candidate_actions",
            "progressive_widening_growth_interval",
            "progressive_widening_growth_base",
            "root_dirichlet_noise_enabled",
            "root_dirichlet_noise_fraction",
            "root_dirichlet_alpha",
            "hidden_prior_mass",
            "fpu_reduction",
            "virtual_loss",
            "mcts_session_cache_max_states",
            "mcts_active_root_limit",
            "max_actions",
            "temperature",
        },
    )
    evaluation = _section(
        config,
        "evaluation",
        {"games_per_epoch", "sealbot_variant", "sealbot_time_limit", "max_actions", "require_sealbot"},
    )
    performance = _section(
        config,
        "performance",
        {
            "calibrate",
            "target_selfplay_positions_per_second",
            "inference_batch_candidates",
            "selfplay_batch_candidates",
            "training_batch_candidates",
            "mcts_virtual_batch_candidates",
            "selfplay_probe_positions",
            "probe_batches",
        },
    )

    return Model1Config(
        architecture=Model1ArchitectureConfig(
            input_channels=_positive_int(arch, "input_channels", INPUT_CHANNELS),
            channels=_positive_int(arch, "channels", DEFAULT_CHANNELS),
            residual_blocks=_positive_int(arch, "residual_blocks", DEFAULT_BLOCKS),
            dropout=_bounded_float(arch, "dropout", 0.0, minimum=0.0, maximum=1.0, include_maximum=False),
            lookahead_horizons=_positive_int_tuple(arch, "lookahead_horizons", ()),
        ),
        training=Model1TrainingConfig(
            batch_size=_positive_int(training, "batch_size", 128),
            learning_rate=_positive_float(training, "learning_rate", 1.0e-3),
            weight_decay=_nonnegative_float(training, "weight_decay", 1.0e-4),
            policy_weight=_nonnegative_float(training, "policy_weight", 1.0),
            value_weight=_nonnegative_float(training, "value_weight", 1.0),
            opp_policy_weight=_nonnegative_float(training, "opp_policy_weight", 0.25),
            lookahead_weight=_nonnegative_float(training, "lookahead_weight", 0.25),
            amp=_bool(training, "amp", True),
            max_grad_norm=_nonnegative_float(training, "max_grad_norm", 1.0),
        ),
        samples=Model1SampleConfig(
            capacity=_minimum_int(samples, "capacity", 200_000, minimum=200_000),
            train_sample_count=_positive_int(samples, "train_sample_count", 4096),
            recency_halflife=_positive_float(samples, "recency_halflife", 50_000.0),
            compression_level=_int_range(samples, "compression_level", 6, minimum=0, maximum=9),
        ),
        selfplay=Model1SelfPlayConfig(
            samples_per_epoch=_positive_int(selfplay, "samples_per_epoch", 4096),
            search_visits=_positive_int(selfplay, "search_visits", 128),
            active_games=_positive_int(selfplay, "active_games", 2048),
            min_mcts_samples_per_game=_positive_int(selfplay, "min_mcts_samples_per_game", 32),
            progressive_widening_initial_actions=_positive_int(selfplay, "progressive_widening_initial_actions", 8),
            progressive_widening_child_initial_actions=_positive_int(selfplay, "progressive_widening_child_initial_actions", 4),
            progressive_widening_candidate_actions=_positive_int(selfplay, "progressive_widening_candidate_actions", 128),
            progressive_widening_growth_interval=_positive_float(selfplay, "progressive_widening_growth_interval", 256.0),
            progressive_widening_growth_base=_greater_than_float(selfplay, "progressive_widening_growth_base", 1.3, minimum=1.0),
            root_dirichlet_noise_enabled=_bool(selfplay, "root_dirichlet_noise_enabled", True),
            root_dirichlet_noise_fraction=_bounded_float(selfplay, "root_dirichlet_noise_fraction", 0.25, minimum=0.0, maximum=1.0),
            root_dirichlet_alpha=_positive_float(selfplay, "root_dirichlet_alpha", 0.03),
            hidden_prior_mass=_bounded_float(selfplay, "hidden_prior_mass", 0.05, minimum=0.0, maximum=0.95),
            fpu_reduction=_nonnegative_float(selfplay, "fpu_reduction", 0.20),
            virtual_loss=_nonnegative_float(selfplay, "virtual_loss", 1.0),
            mcts_session_cache_max_states=_positive_int(selfplay, "mcts_session_cache_max_states", 1_048_576),
            mcts_active_root_limit=_positive_int(selfplay, "mcts_active_root_limit", 1024),
            max_actions=_positive_int(selfplay, "max_actions", 1024),
            temperature=_nonnegative_float(selfplay, "temperature", 1.0),
        ),
        evaluation=Model1EvalConfig(
            games_per_epoch=_nonnegative_int(evaluation, "games_per_epoch", 64),
            sealbot_variant=str(evaluation.get("sealbot_variant", "best")),
            sealbot_time_limit=_positive_float(evaluation, "sealbot_time_limit", 0.05),
            max_actions=_positive_int(evaluation, "max_actions", 1024),
            require_sealbot=_bool(evaluation, "require_sealbot", False),
        ),
        performance=Model1PerformanceConfig(
            calibrate=_bool(performance, "calibrate", True),
            target_selfplay_positions_per_second=_positive_float(performance, "target_selfplay_positions_per_second", 128.0),
            inference_batch_candidates=_positive_int_tuple(performance, "inference_batch_candidates", (128, 256, 512, 1024), non_empty=True),
            selfplay_batch_candidates=_positive_int_tuple(performance, "selfplay_batch_candidates", (2048,), non_empty=True),
            training_batch_candidates=_positive_int_tuple(performance, "training_batch_candidates", (64, 128, 192, 256), non_empty=True),
            mcts_virtual_batch_candidates=_positive_int_tuple(performance, "mcts_virtual_batch_candidates", (4,), non_empty=True),
            selfplay_probe_positions=_positive_int(performance, "selfplay_probe_positions", 8192),
            probe_batches=_positive_int(performance, "probe_batches", 1),
        ),
        device=str(config.get("device", "cuda")),
        checkpoint_path=_optional_path(config.get("checkpoint_path")),
    )


def _section(raw: Mapping[str, Any], name: str, allowed: set[str]) -> Mapping[str, Any]:
    """Return a typed config subsection after rejecting unsupported keys."""

    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"model config section {name!r} must be a mapping")
    _reject_unknown(value, f"model config section {name!r}", allowed)
    return value


def _optional_path(value: object) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _reject_unknown(raw: Mapping[str, Any], label: str, allowed: set[str]) -> None:
    """Fail fast when config still contains removed or misspelled options."""

    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported key(s): {', '.join(unknown)}")


def _bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _int_value(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if resolved != value and not isinstance(value, str):
        raise ValueError(f"{key} must be an integer")
    return resolved


def _nonnegative_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = _int_value(raw, key, default)
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _positive_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = _int_value(raw, key, default)
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _minimum_int(raw: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    value = _int_value(raw, key, default)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _int_range(raw: Mapping[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = _int_value(raw, key, default)
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be in [{minimum}, {maximum}]")
    return value


def _float_value(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{key} must be finite")
    return resolved


def _positive_float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = _float_value(raw, key, default)
    if value <= 0.0:
        raise ValueError(f"{key} must be > 0")
    return value


def _greater_than_float(raw: Mapping[str, Any], key: str, default: float, *, minimum: float) -> float:
    value = _float_value(raw, key, default)
    if value <= minimum:
        raise ValueError(f"{key} must be > {minimum}")
    return value


def _nonnegative_float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = _float_value(raw, key, default)
    if value < 0.0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _bounded_float(
    raw: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    include_maximum: bool = True,
) -> float:
    value = _float_value(raw, key, default)
    above_minimum = value >= minimum
    below_maximum = value <= maximum if include_maximum else value < maximum
    if not above_minimum or not below_maximum:
        right = "]" if include_maximum else ")"
        raise ValueError(f"{key} must be in [{minimum}, {maximum}{right}")
    return value


def _positive_int_tuple(
    raw: Mapping[str, Any],
    key: str,
    default: tuple[int, ...],
    *,
    non_empty: bool = False,
) -> tuple[int, ...]:
    value = raw.get(key, default)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, tuple | list):
        raise ValueError(f"{key} must be a sequence of positive integers")
    result = tuple(_coerce_positive_int_item(key, item) for item in value)
    if non_empty and not result:
        raise ValueError(f"{key} must not be empty")
    return result


def _coerce_positive_int_item(key: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must contain only positive integers")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain only positive integers") from exc
    if resolved <= 0:
        raise ValueError(f"{key} must contain only positive integers")
    return resolved
