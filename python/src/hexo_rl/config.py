"""Configuration loading for Hexo RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


@dataclass(frozen=True)
class GameConfig:
    name: str = "hexo"
    crop_size: int = 31
    rules_version: int = 1
    max_placements: int = 300


@dataclass(frozen=True)
class ModelConfig:
    package: str = "hexo_resnet"
    variant: str = "small"
    channels: int = 64
    residual_blocks: int = 6
    precision: str = "fp16"


@dataclass(frozen=True)
class SelfplayConfig:
    games_per_cycle: int = 200
    actors: int = 8
    mcts_visits: int = 64
    temperature_placements: int = 24
    dirichlet_noise: bool = True
    rust_binary: str | None = None


@dataclass(frozen=True)
class MctsConfig:
    c_puct: float = 1.5
    root_noise_alpha: float = 0.3
    root_noise_frac: float = 0.25


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 64
    device: str = "cuda"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    steps_per_cycle: int = 500
    replay_window_samples: int = 100_000
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    amp: bool = True
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class CheckpointingConfig:
    latest_path: str = "data/checkpoints/latest.pt"
    keep_last: int = 10


@dataclass(frozen=True)
class LoopConfig:
    cycles: int = 1000


@dataclass(frozen=True)
class PathsConfig:
    replay_latest: str = "data/replay/replay_latest.jsonl"
    metrics_log: str = "data/metrics.jsonl"
    selfplay_root: str = "data/selfplay"


@dataclass(frozen=True)
class HexoConfig:
    game: GameConfig = field(default_factory=GameConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    root: Path = field(default_factory=lambda: Path.cwd())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        return data


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], raw: Mapping[str, Any] | None) -> T:
    if raw is None:
        raw = {}
    valid = {item.name for item in fields(cls)}
    kwargs = {key: value for key, value in raw.items() if key in valid}
    return cls(**kwargs)  # type: ignore[misc]


def load_config(path: str | Path) -> HexoConfig:
    """Load a YAML config file into typed dataclasses."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Config {config_path} must contain a YAML mapping")

    root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return HexoConfig(
        game=_coerce_dataclass(GameConfig, raw.get("game")),
        model=_coerce_dataclass(ModelConfig, raw.get("model")),
        selfplay=_coerce_dataclass(SelfplayConfig, raw.get("selfplay")),
        mcts=_coerce_dataclass(MctsConfig, raw.get("mcts")),
        inference=_coerce_dataclass(InferenceConfig, raw.get("inference")),
        training=_coerce_dataclass(TrainingConfig, raw.get("training")),
        checkpointing=_coerce_dataclass(CheckpointingConfig, raw.get("checkpointing")),
        loop=_coerce_dataclass(LoopConfig, raw.get("loop")),
        paths=_coerce_dataclass(PathsConfig, raw.get("paths")),
        root=root,
    )


def resolve_path(config: HexoConfig, path: str | Path) -> Path:
    """Resolve project-relative paths against the config root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config.root / candidate).resolve()

