"""ResNet sample writing and finalization boundary.

During self-play, the ResNet player should record the data needed to train
ResNet later: encoded input references, legal action order, policy/search
output, selected action, and enough metadata to finalize value after game end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import ResNetSampleSettings


@dataclass(slots=True)
class ResNetSampleFinalizer:
    """Placeholder for finalizing pending ResNet samples after games end."""

    config: ResNetSampleSettings = field(default_factory=ResNetSampleSettings)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def finalize(self, *, ctx: Any, components: Any) -> Mapping[str, Any]:
        """Finalize result-dependent targets such as scalar value."""

        _ = components
        return {
            "status": "skipped",
            "reason": "ResNet sample finalization is not implemented yet.",
            "run": ctx.config.run.name,
        }
