"""Training config loading and normalization.

This module should eventually become the typed contract for training runs.
For now it accepts simple YAML/TOML dictionaries and normalizes the fields
needed by the orchestration skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import tomllib


ConfigMap = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model plugin selection plus model-owned config."""

    name: str
    module: str | None = None
    entry_point: str | None = None
    config: ConfigMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Shared run identity and output locations."""

    name: str = "hexo_train_run"
    output_dir: Path = Path("runs/hexo_train_run")
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Normalized config consumed by `TrainingPipeline`."""

    model: ModelConfig
    stages: tuple[str, ...] = ("all",)
    run: RunConfig = field(default_factory=RunConfig)
    shared: ConfigMap = field(default_factory=dict)
    model_specific: ConfigMap = field(default_factory=dict)
    raw: ConfigMap = field(default_factory=dict)


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load a YAML or TOML config and return a normalized training config."""

    path = Path(config_path)
    raw = _load_raw_config(path)
    return normalize_training_config(raw, base_dir=path.parent)


def normalize_training_config(raw: ConfigMap, *, base_dir: Path) -> TrainingConfig:
    """Convert an untyped config mapping into the training skeleton contract."""

    model_section = _require_mapping(raw, "model")
    model_name = str(model_section.get("name", "")).strip()
    if not model_name:
        raise ValueError("Training config must define model.name.")

    stage_values = raw.get("stages", ("all",))
    if isinstance(stage_values, str):
        stages = (stage_values,)
    else:
        stages = tuple(str(stage) for stage in stage_values)
    if not stages:
        raise ValueError("Training config must define at least one stage.")

    run_section = dict(raw.get("run", {}))
    run_name = str(run_section.get("name", f"{model_name}_train"))
    output_dir = Path(run_section.get("output_dir", Path("runs") / run_name))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    model = ModelConfig(
        name=model_name,
        module=_optional_str(model_section.get("module")),
        entry_point=_optional_str(model_section.get("entry_point")),
        config=dict(model_section.get("config", {})),
    )
    run = RunConfig(
        name=run_name,
        output_dir=output_dir,
        seed=_optional_int(run_section.get("seed")),
    )

    return TrainingConfig(
        model=model,
        stages=stages,
        run=run,
        shared=dict(raw.get("shared", {})),
        model_specific=dict(raw.get("model_specific", {})),
        raw=dict(raw),
    )


def _load_raw_config(path: Path) -> ConfigMap:
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError("YAML training configs require PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("YAML training config must load to a mapping.")
        return loaded
    raise ValueError(f"Unsupported training config format: {path.suffix}")


def _require_mapping(raw: ConfigMap, key: str) -> ConfigMap:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config must define a [{key}] mapping.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
