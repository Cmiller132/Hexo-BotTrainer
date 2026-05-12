"""Inference adapter for the ResNet model family.

The runner should interact with a player adapter, not raw tensors. This module
keeps tensor execution, policy/value decoding, and device concerns inside the
model package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn

import torch


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Decoded policy/value output from the ResNet model."""

    policy_logits: torch.Tensor
    value: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _not_implemented(operation: str) -> NoReturn:
    raise NotImplementedError(f"{operation} will be backed by ResNet inference.")


class ResNetInferenceAdapter:
    """Thin wrapper around a `HexoNet` instance."""

    def __init__(self, model: torch.nn.Module, *, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)

    def infer(self, model_input: object) -> InferenceResult:
        """Run the model and decode outputs for a decision."""

        _not_implemented("ResNetInferenceAdapter.infer")
