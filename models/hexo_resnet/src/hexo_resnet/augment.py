"""Crop-level augmentation helpers."""

from __future__ import annotations

import random
from typing import Mapping

import torch


SPATIAL_KEYS = {"state_tensor", "legal_mask", "policy_target"}


def _transform_tensor(tensor: torch.Tensor, transform: int) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    if transform >= 4:
        tensor = torch.flip(tensor, dims=(-1,))
        transform -= 4
    if transform:
        tensor = torch.rot90(tensor, k=transform, dims=(-2, -1))
    return tensor


def augment_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply one crop symmetry to state, legal mask, and policy target.

    This is a square-crop approximation. The Rust encoder can later expose exact
    axial-coordinate symmetries while keeping this model API unchanged.
    """

    transform = random.randrange(8)
    augmented: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        augmented[key] = _transform_tensor(value, transform) if key in SPATIAL_KEYS else value
    return augmented

