"""Dynamic model plugin loading.

Model packages provide the model-specific pieces: architecture construction,
sample decoding, losses, checkpoint interpretation, and optional stage hooks.
`hexo_train` only finds the plugin and calls agreed-upon lifecycle methods.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from .config import ModelConfig
from .components import ComponentOverrides, DefaultTrainingComponents, SharedComponents


@runtime_checkable
class ModelPlugin(Protocol):
    """Minimum plugin shape expected by the training orchestrator."""

    name: str

    def build_model(self, game_spec: Any, config: Any) -> Any:
        """Build and return the model object."""

    def training_component_overrides(
        self,
        *,
        defaults: DefaultTrainingComponents,
        config: Any,
        shared: SharedComponents,
    ) -> ComponentOverrides | None:
        """Return only the default components this model wants to replace."""


def load_model_plugin(config: ModelConfig) -> ModelPlugin:
    """Load a model plugin by explicit module, entry point, or plugin name."""

    if config.module:
        return _load_from_module(config.module)
    if config.entry_point:
        return _load_from_entry_point(config.entry_point)
    return _load_by_name(config.name)


def _load_from_module(module_name: str) -> ModelPlugin:
    module = import_module(module_name)
    if hasattr(module, "get_plugin"):
        return module.get_plugin()
    if hasattr(module, "plugin"):
        return module.plugin
    raise AttributeError(
        f"Model module {module_name!r} must expose get_plugin() or plugin."
    )


def _load_from_entry_point(entry_point_name: str) -> ModelPlugin:
    for group in _entry_point_groups():
        for entry_point in entry_points(group=group):
            if entry_point.name == entry_point_name:
                return entry_point.load()()
    raise LookupError(f"No Hexo model entry point named {entry_point_name!r}.")


def _load_by_name(model_name: str) -> ModelPlugin:
    for group in _entry_point_groups():
        for entry_point in entry_points(group=group):
            if entry_point.name == model_name:
                loaded = entry_point.load()
                return loaded() if callable(loaded) else loaded

    # Development fallback: `hexo_model_resnet` can be used before packaging.
    return _load_from_module(model_name)


def _entry_point_groups() -> tuple[str, ...]:
    return (
        "hexo_train.models",
        "hexo_utils.models",
    )
