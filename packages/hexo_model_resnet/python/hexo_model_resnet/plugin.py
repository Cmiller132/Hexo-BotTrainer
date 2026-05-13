"""Model plugin entry point."""

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

        Pseudocode shape:

        - replace sample decoding with ResNet crop/input decoding;
        - replace training with the ResNet optimizer/loss loop.
        - rely on `hexo_train` to keep shared defaults when not overridden.
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
    return plugin
