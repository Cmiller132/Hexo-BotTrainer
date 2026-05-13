"""Configuration for the ResNet model family.

The model package owns architecture and training defaults. Runner config should
select or pass this configuration, but it should not know the meaning of ResNet
channels, blocks, or tensor shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ResNetArchitectureConfig:
    """Model construction settings for `architecture.HexoNet`."""

    input_channels: int = 12
    channels: int = 64
    residual_blocks: int = 6
    crop_size: int = 15


@dataclass(frozen=True, slots=True)
class ResNetTrainingSettings:
    """Training settings owned by this model family."""

    batch_size: int = 128
    learning_rate: float = 1.0e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ResNetSampleSettings:
    """Sample decoding/finalization settings owned by this model family."""

    crop_size: int = 15
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResNetConfig:
    """Complete ResNet-owned config parsed from `[model.config]`."""

    architecture: ResNetArchitectureConfig = field(default_factory=ResNetArchitectureConfig)
    training: ResNetTrainingSettings = field(default_factory=ResNetTrainingSettings)
    samples: ResNetSampleSettings = field(default_factory=ResNetSampleSettings)


def parse_resnet_config(raw: Mapping[str, Any] | None) -> ResNetConfig:
    """Parse the ResNet package config without leaking fields into `hexo_train`."""

    config = dict(raw or {})
    architecture = dict(config.get("architecture", config))
    training = dict(config.get("training", {}))
    samples = dict(config.get("samples", {}))

    return ResNetConfig(
        architecture=ResNetArchitectureConfig(
            input_channels=int(architecture.get("input_channels", 12)),
            channels=int(architecture.get("channels", 64)),
            residual_blocks=int(architecture.get("residual_blocks", 6)),
            crop_size=int(architecture.get("crop_size", samples.get("crop_size", 15))),
        ),
        training=ResNetTrainingSettings(
            batch_size=int(training.get("batch_size", 128)),
            learning_rate=float(training.get("learning_rate", 1.0e-3)),
            policy_weight=float(training.get("policy_weight", 1.0)),
            value_weight=float(training.get("value_weight", 1.0)),
        ),
        samples=ResNetSampleSettings(
            crop_size=int(samples.get("crop_size", architecture.get("crop_size", 15))),
            metadata=dict(samples.get("metadata", {})),
        ),
    )
