"""Training step stage.

This stage owns lifecycle concerns such as how many steps to ask for and where
metrics are reported. The model trainer owns tensor decoding, loss computation,
optimizer state, and model-specific metrics.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def train_steps(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Run configured training steps through the model-owned trainer."""

    train_config = ctx.section("train")
    steps = int(train_config.get("steps", 0))
    trainer = components.model.trainer

    if trainer is not None and hasattr(trainer, "train_steps"):
        return trainer.train_steps(
            steps=steps,
            sample_window=components.shared.sample_window,
            ctx=ctx,
            components=components,
        )

    return {
        "status": "skipped",
        "steps": steps,
        "reason": "model trainer not wired yet",
    }
