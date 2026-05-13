"""ResNet trainer boundary.

`hexo_train` decides when training runs and how many passes are requested. This
module owns ResNet-specific decoding, forward passes, loss computation,
optimizer steps, D6 application through the decoder, and metrics for each pass.

The implementation is still a placeholder, but the method shape is the real
contract: `hexo_train` passes selected samples and symmetries in, and ResNet
will eventually turn those into tensors and optimizer updates here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import ResNetTrainingSettings


@dataclass(slots=True)
class ResNetTrainer:
    """Placeholder ResNet train-pass implementation.

    When real training is wired in, this object should keep ResNet-specific
    optimizer state and use the model-owned decoder/loss helpers.
    """

    model: Any | None = None
    config: ResNetTrainingSettings = field(default_factory=ResNetTrainingSettings)

    def train_passes(
        self,
        *,
        passes: int,
        sample_window: object,
        sample_symmetries: object,
        ctx: Any,
        components: Any,
        epoch: int,
    ) -> Mapping[str, Any]:
        """Run model-owned training passes once tensor code exists.

        Expected future flow:

        1. Decode `sample_window` into batches.
        2. Apply `sample_symmetries` consistently to inputs and targets.
        3. Run forward/loss/backward/optimizer work for `passes`.
        4. Return metrics for diagnostics.
        """

        _ = (sample_window, sample_symmetries, components)
        return {
            "status": "skipped",
            "epoch": epoch,
            "passes": passes,
            "reason": "ResNet trainer is not implemented yet.",
            "run": ctx.config.run.name,
            "has_model": self.model is not None,
        }
