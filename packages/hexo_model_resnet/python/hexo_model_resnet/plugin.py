"""Model plugin entry point.

This file is the composition root for the ResNet model package. `hexo_train`
loads this plugin dynamically, then asks it to build the model and return the
model-owned training components that replace or extend shared defaults.

Keep detailed architecture, decoding, loss, augmentation, and training logic in
their own modules. The plugin should mostly wire those pieces together.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .architecture import HexoNet
from .config import parse_resnet_config
from .decode import ResNetSampleDecoder
from .samples import ResNetSampleFinalizer
from .trainer import ResNetTrainer


class HexoResNetPlugin:
    """Training plugin consumed by `hexo_train`."""

    name = "hexo_model_resnet"

    def training_component_overrides(
        self,
        *,
        defaults: Any,
        config: Mapping[str, Any],
        shared: Any,
        model: torch.nn.Module | None,
    ) -> ComponentOverrides:
        """Describe which shared training defaults ResNet accepts or replaces.

        Step by step:

        1. Parse only the ResNet-owned portion of config.
        2. Build ResNet-specific sample decoding/finalization/training helpers.
        3. Return only those replacements; `hexo_train` keeps shared defaults
           for everything else.
        """

        _ = (defaults, shared)
        resnet_config = parse_resnet_config(config)
        return ComponentOverrides(
            sample_decoder=ResNetSampleDecoder(config=resnet_config.samples),
            sample_finalizer=ResNetSampleFinalizer(config=resnet_config.samples),
            trainer=ResNetTrainer(model=model, config=resnet_config.training),
            extra={
                "sample_decoder": "ResNet crop/input decoder placeholder.",
                "sample_finalizer": "ResNet value finalizer placeholder.",
                "trainer": "ResNet training loop placeholder.",
            },
        )

    def build_model(
        self,
        game_spec: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Build the ResNet model from model-owned architecture settings."""

        _ = game_spec
        resnet_config = parse_resnet_config(config)
        architecture = resnet_config.architecture
        return HexoNet(
            in_channels=architecture.input_channels,
            channels=architecture.channels,
            blocks=architecture.residual_blocks,
        )


plugin = HexoResNetPlugin()


def get_plugin() -> HexoResNetPlugin:
    """Return the singleton plugin used by entry point and module loading."""

    return plugin
