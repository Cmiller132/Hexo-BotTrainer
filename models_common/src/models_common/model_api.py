"""Stable API implemented by model packages."""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

TensorBatch = Mapping[str, torch.Tensor]


@runtime_checkable
class ModelPlugin(Protocol):
    """Protocol shared by the runner and independently packaged models."""

    name: str

    def build_model(
        self,
        game_spec: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Create an untrained model for the given game and model config."""

    def forward_inference(
        self,
        model: torch.nn.Module,
        batch: TensorBatch,
    ) -> Mapping[str, torch.Tensor]:
        """Return policy_logits [B, H, W] and value [B]."""

    def loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: TensorBatch,
    ) -> torch.Tensor:
        """Return the scalar training loss for one batch."""

    def augment_batch(self, batch: TensorBatch) -> TensorBatch:
        """Apply model-owned training augmentation."""


def load_model_plugin(package: str) -> ModelPlugin:
    """Import a model package and return its plugin object.

    Packages may expose either ``get_plugin()`` or a module-level ``plugin``.
    """

    module = importlib.import_module(package)
    if hasattr(module, "get_plugin"):
        plugin = module.get_plugin()
    elif hasattr(module, "plugin"):
        plugin = module.plugin
    else:
        raise AttributeError(
            f"Model package {package!r} must expose get_plugin() or plugin"
        )

    if not isinstance(plugin, ModelPlugin):
        missing = [
            name
            for name in (
                "name",
                "build_model",
                "forward_inference",
                "loss",
                "augment_batch",
            )
            if not hasattr(plugin, name)
        ]
        if missing:
            raise TypeError(
                f"Model plugin from {package!r} is missing: {', '.join(missing)}"
            )
    return plugin

