"""ResNet trainer boundary.

`hexo_train` decides when training runs and how many steps are requested. This
module owns ResNet-specific decoding, forward passes, loss computation,
optimizer steps, and metrics for each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ResNetTrainer:
    """Placeholder ResNet train-step implementation."""

    config: Mapping[str, Any] = field(default_factory=dict)

    def train_steps(
        self,
        *,
        steps: int,
        sample_window: object,
        ctx: Any,
        components: Any,
    ) -> Mapping[str, Any]:
        """Run model-owned training steps once tensor code exists."""

        _ = (sample_window, components)
        return {
            "status": "skipped",
            "steps": steps,
            "reason": "ResNet trainer is not implemented yet.",
            "run": ctx.config.run.name,
        }
