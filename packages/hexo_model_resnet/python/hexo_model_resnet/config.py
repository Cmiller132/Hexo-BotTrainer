"""Configuration for the ResNet model family.

The model package owns architecture and training defaults. Runner config should
select or pass this configuration, but it should not know the meaning of ResNet
channels, blocks, or tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResNetConfig:
    """Model construction settings for `architecture.HexoNet`."""

    input_channels: int = 12
    channels: int = 64
    residual_blocks: int = 6
    crop_size: int = 15


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training settings owned by this model family."""

    batch_size: int = 128
    learning_rate: float = 1.0e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
