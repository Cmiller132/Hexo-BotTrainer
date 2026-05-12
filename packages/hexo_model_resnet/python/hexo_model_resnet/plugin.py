"""Model plugin entry point."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .augment import augment_batch
from .losses import hexo_loss
from .architecture import HexoNet


class HexoResNetPlugin:
    name = "hexo_model_resnet"

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
