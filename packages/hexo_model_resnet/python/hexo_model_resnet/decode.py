"""ResNet sample decoding boundary.

Shared sample utilities select records and symmetries. This module will turn
those selected records into ResNet tensors: board planes, scalar globals,
legal masks, policy targets, values, and any model-specific weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ResNetSampleDecoder:
    """Placeholder decoder for `hexo_utils.samples.SampleBatch`."""

    config: Mapping[str, Any] = field(default_factory=dict)

    def decode(self, sample_batch: object) -> Mapping[str, Any]:
        """Convert sampled records into tensors once tensor code exists."""

        _ = sample_batch
        raise NotImplementedError("ResNet sample decoding is not implemented yet.")
