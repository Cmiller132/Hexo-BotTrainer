"""Self-play generation stage.

The runner executes games. Model packages provide model-backed players and
sample writers. `hexo_train` decides whether self-play happens in this run and
where its outputs fit in the training sequence.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def maybe_generate_selfplay(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Generate self-play samples when enabled by config."""

    selfplay_config = ctx.section("selfplay")
    if not bool(selfplay_config.get("enabled", False)):
        return {"status": "skipped", "reason": "selfplay disabled"}

    if hasattr(components.model.plugin, "build_selfplay_request"):
        request = components.model.plugin.build_selfplay_request(ctx, components)
    else:
        request = {
            "config": dict(selfplay_config),
            "sample_store": components.shared.sample_store,
            "note": "Runner self-play wiring is not implemented yet.",
        }

    result = {
        "status": "planned",
        "request": request,
        "note": "Future implementation should call hexo_runner self-play.",
    }
    components.shared.selfplay_result = result
    return result
