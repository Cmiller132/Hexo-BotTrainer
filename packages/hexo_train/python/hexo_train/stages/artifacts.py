"""Run artifact and diagnostics stages."""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def write_diagnostics(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Write a final human-readable summary of the training run."""

    summary = {
        "run": ctx.config.run.name,
        "model": ctx.config.model.name,
        "output_dir": str(ctx.output_dir),
        "checkpoint_dir": str(ctx.checkpoint_dir),
        "samples_dir": str(ctx.samples_dir),
        "completed_stages": list(ctx.stage_outputs),
        "model_extra": dict(components.model.extra),
    }
    ctx.diagnostics.write_json("run.completed.json", summary)
    return summary
