"""ResNet trainer boundary.

`hexo_train` decides when training runs and how many steps are requested. This
module owns ResNet-specific decoding, forward passes, loss computation,
optimizer steps, D6 application through the decoder, and metrics for each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ResNetTrainingSettings


@dataclass(slots=True)
class ResNetTrainer:
    """Placeholder ResNet train-step implementation."""

    model: Any | None = None
    config: ResNetTrainingSettings = field(default_factory=ResNetTrainingSettings)

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
            "has_model": self.model is not None,
        }
