"""Sample symmetry selection stage."""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def select_sample_symmetries(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Attach deterministic D6 choices to the current sample window."""

    train_config = ctx.section("train")
    epoch = int(train_config.get("epoch", 0))
    selector = components.model.symmetry_selector or components.shared.defaults.symmetry_selector
    selection = selector.select_for_window(
        components.shared.sample_window,
        seed=ctx.config.run.seed,
        epoch=epoch,
    )
    components.shared.sample_symmetries = selection
    return {
        "symmetry_count": len(selection.symmetries),
        "seed": selection.seed,
        "epoch": selection.epoch,
        "metadata": dict(selection.metadata),
    }
