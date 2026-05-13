"""Canonical training pipeline orchestration.

`hexo_train` owns the order of a training run. Model packages can override the
behavior inside a stage, but they should not have to recreate the lifecycle.

The default sequence is:

1. load and validate config;
2. create run context;
3. load model training plugin;
4. build default and model-specific components;
5. load or initialize checkpoint;
6. prepare sample store;
7. optionally generate self-play samples;
8. finalize pending samples;
9. refresh sample index;
10. build sample window;
11. select sample symmetries;
12. train configured steps;
13. save checkpoint;
14. optionally update self-play checkpoint pointer;
15. write diagnostics.

The model owns tensors, targets, losses, optimizer details, and model artifact
contents. Utilities own reusable sample mechanics. Runner owns game execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .components import (
    TrainingComponents,
    build_model_components,
)
from .config import load_training_config
from .context import RunContext
from .defaults import build_shared_components
from .registry import load_model_plugin
from .stages import (
    build_sample_window,
    finalize_pending_samples,
    load_or_initialize_checkpoint,
    maybe_generate_selfplay,
    prepare_sample_store,
    refresh_sample_index,
    save_checkpoint,
    select_sample_symmetries,
    train_steps,
    update_selfplay_checkpoint_pointer,
    write_diagnostics,
)


StageFunction = Callable[[RunContext, TrainingComponents], Any]


CANONICAL_STAGE_ORDER: tuple[tuple[str, StageFunction], ...] = (
    ("load_or_initialize_checkpoint", load_or_initialize_checkpoint),
    ("prepare_sample_store", prepare_sample_store),
    ("maybe_generate_selfplay", maybe_generate_selfplay),
    ("finalize_pending_samples", finalize_pending_samples),
    ("refresh_sample_index", refresh_sample_index),
    ("build_sample_window", build_sample_window),
    ("select_sample_symmetries", select_sample_symmetries),
    ("train_steps", train_steps),
    ("save_checkpoint", save_checkpoint),
    ("update_selfplay_checkpoint_pointer", update_selfplay_checkpoint_pointer),
    ("write_diagnostics", write_diagnostics),
)


class TrainingPipeline:
    """Shared orchestrator for model training runs."""

    def run(self, config_path: str | Path) -> RunContext:
        """Run the canonical training lifecycle from a YAML/TOML config path."""

        config = load_training_config(config_path)
        ctx = RunContext.from_config(config)
        plugin = load_model_plugin(config.model)
        shared = build_shared_components(ctx)
        model = build_model_components(plugin=plugin, ctx=ctx, shared=shared)
        components = TrainingComponents(shared=shared, model=model)

        ctx.diagnostics.write_json("config.normalized.json", config)
        for stage_name, default_stage in self._selected_stages(ctx):
            self._run_stage(stage_name, default_stage, ctx, components)
        return ctx

    def _selected_stages(
        self,
        ctx: RunContext,
    ) -> tuple[tuple[str, StageFunction], ...]:
        """Return configured stages in canonical order.

        Config may select a subset by name, but it does not define ordering.
        This prevents training lifecycle policy from drifting into experiment
        configs.
        """

        requested = set(ctx.config.stages)
        if not requested or requested == {"all"}:
            return CANONICAL_STAGE_ORDER
        known = {name for name, _ in CANONICAL_STAGE_ORDER}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown training stages: {sorted(unknown)}")
        return tuple(
            (name, stage)
            for name, stage in CANONICAL_STAGE_ORDER
            if name in requested
        )

    def _run_stage(
        self,
        stage_name: str,
        default_stage: StageFunction,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Any:
        """Run one stage with diagnostics and optional model override."""

        diagnostics = ctx.diagnostics
        started_at = diagnostics.start_stage(stage_name)
        try:
            result = self._dispatch_stage(
                stage_name,
                default_stage,
                ctx,
                components,
            )
        except Exception as exc:
            diagnostics.finish_stage(
                stage=stage_name,
                started_at=started_at,
                status="failed",
                metadata={"error": repr(exc)},
            )
            raise

        ctx.remember(stage_name, result)
        diagnostics.finish_stage(
            stage=stage_name,
            started_at=started_at,
            status=self._status_for(result),
            metadata={"result": self._result_metadata(result)},
        )
        return result

    def _dispatch_stage(
        self,
        stage_name: str,
        default_stage: StageFunction,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Any:
        """Call a model override when present, otherwise the shared default."""

        handler = components.model.stage_handlers.get(stage_name)
        if handler is not None:
            return handler(ctx, components)

        plugin = components.model.plugin
        if hasattr(plugin, stage_name):
            return getattr(plugin, stage_name)(ctx, components)

        return default_stage(ctx, components)

    def _status_for(self, result: Any) -> str:
        if isinstance(result, Mapping) and result.get("status") == "skipped":
            return "skipped"
        return "completed"

    def _result_metadata(self, result: Any) -> Mapping[str, Any]:
        if result is None:
            return {}
        if isinstance(result, Mapping):
            return result
        if isinstance(result, Path):
            return {"path": str(result)}
        return {"type": type(result).__name__, "repr": repr(result)}
