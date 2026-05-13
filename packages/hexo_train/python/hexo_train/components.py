"""Training component contracts.

The training pipeline has three layers:

1. `RunContext` owns run identity, directories, diagnostics, and stage outputs.
2. `SharedComponents` are reusable defaults created by `hexo_train`.
3. `ModelComponents` are supplied or overridden by the selected model plugin.

Stages receive `RunContext` plus `TrainingComponents`. This keeps the stage
sequence centralized while still letting models own tensors, losses, decoders,
and checkpoint contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .context import RunContext


StageHandler = Callable[[RunContext, "TrainingComponents"], Any]


@dataclass(slots=True)
class DefaultTrainingComponents:
    """Shared defaults a model may accept or override."""

    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    symmetry_selector: Any | None = None
    checkpoint_store: Any | None = None
    diagnostics: Any | None = None


@dataclass(slots=True)
class SharedComponents:
    """Model-neutral handles built by `hexo_train`."""

    defaults: DefaultTrainingComponents
    sample_store: Any | None = None
    sample_index: Any | None = None
    sample_window: Any | None = None
    sample_symmetries: Any | None = None
    checkpoint_state: Any | None = None
    selfplay_result: Any | None = None
    game_spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentOverrides:
    """Optional replacements returned by a model training plugin."""

    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    sample_decoder: Any | None = None
    sample_finalizer: Any | None = None
    symmetry_selector: Any | None = None
    trainer: Any | None = None
    optimizer: Any | None = None
    checkpoint_loader: Any | None = None
    checkpoint_saver: Any | None = None
    stage_handlers: Mapping[str, StageHandler] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelComponents:
    """Model-owned training pieces used by generic stages."""

    plugin: Any
    model: Any | None = None
    optimizer: Any | None = None
    trainer: Any | None = None
    decoder: Any | None = None
    sample_finalizer: Any | None = None
    symmetry_selector: Any | None = None
    checkpoint_loader: Any | None = None
    checkpoint_saver: Any | None = None
    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    stage_handlers: Mapping[str, StageHandler] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingComponents:
    """All components available to one pipeline run."""

    shared: SharedComponents
    model: ModelComponents


def build_model_components(
    *,
    plugin: Any,
    ctx: RunContext,
    shared: SharedComponents,
) -> ModelComponents:
    """Build model components by merging shared defaults with plugin overrides."""

    defaults = shared.defaults
    overrides = ComponentOverrides()

    model = None
    if hasattr(plugin, "build_model"):
        model = plugin.build_model(shared.game_spec, ctx.config.model.config)

    if hasattr(plugin, "training_component_overrides"):
        raw_overrides = plugin.training_component_overrides(
            defaults=defaults,
            config=ctx.config.model.config,
            shared=shared,
            model=model,
        )
        overrides = _coerce_overrides(raw_overrides)

    return ModelComponents(
        plugin=plugin,
        model=model,
        optimizer=overrides.optimizer,
        trainer=overrides.trainer,
        decoder=overrides.sample_decoder,
        sample_finalizer=overrides.sample_finalizer,
        symmetry_selector=overrides.symmetry_selector
        or defaults.symmetry_selector,
        checkpoint_loader=overrides.checkpoint_loader,
        checkpoint_saver=overrides.checkpoint_saver,
        scalar_value_target=overrides.scalar_value_target
        or defaults.scalar_value_target,
        legal_policy_target=overrides.legal_policy_target
        or defaults.legal_policy_target,
        stage_handlers=overrides.stage_handlers,
        extra=overrides.extra,
    )


def _coerce_overrides(raw: Any) -> ComponentOverrides:
    if raw is None:
        return ComponentOverrides()
    if isinstance(raw, ComponentOverrides):
        return raw
    if isinstance(raw, Mapping):
        return ComponentOverrides(**raw)
    raise TypeError(
        "training_component_overrides() must return ComponentOverrides, "
        "a mapping, or None."
    )
