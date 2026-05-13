"""Model plugin entry point."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .augment import augment_batch
from .losses import hexo_loss
from .architecture import HexoNet
from .decode import ResNetSampleDecoder
from .samples import ResNetSampleFinalizer
from .trainer import ResNetTrainer


class HexoResNetPlugin:
    name = "hexo_model_resnet"

    def training_component_overrides(
        self,
        *,
        defaults: Any,
        config: Any,
        shared: Any,
    ) -> Mapping[str, Any]:
        """Describe which shared training defaults ResNet accepts or replaces.

        Pseudocode shape:

        - use the default scalar value helper for win/loss/draw targets;
        - use the default legal-action policy helper while policy targets are
          a distribution over engine-provided legal moves;
        - replace sample decoding with ResNet crop/input decoding;
        - replace training with the ResNet optimizer/loss loop.
        """

        return {
            "scalar_value_target": defaults.scalar_value_target,
            "legal_policy_target": defaults.legal_policy_target,
            "sample_decoder": ResNetSampleDecoder(config=getattr(config, "model_specific", {})),
            "sample_finalizer": ResNetSampleFinalizer(),
            "trainer": ResNetTrainer(config=getattr(config, "model_specific", {})),
            "extra": {
                "sample_decoder": "ResNet crop/input decoder placeholder.",
                "sample_finalizer": "ResNet value finalizer placeholder.",
                "trainer": "ResNet training loop placeholder.",
            },
        }

    def build_model(
        self,
        game_spec: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> torch.nn.Module:
        return HexoNet(
            in_channels=int(config.get("input_channels", 12)),
            channels=int(config.get("channels", 64)),
            blocks=int(config.get("residual_blocks", 6)),
        )

    def forward_inference(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        return model(batch["state_tensor"], batch.get("legal_mask"))

    def loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return hexo_loss(outputs, batch)

    def augment_batch(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        return augment_batch(batch)


plugin = HexoResNetPlugin()


def get_plugin() -> HexoResNetPlugin:
    return plugin
