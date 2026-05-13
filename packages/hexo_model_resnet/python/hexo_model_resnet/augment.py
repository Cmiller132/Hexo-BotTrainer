"""Crop-level augmentation helpers."""

from __future__ import annotations

from typing import Mapping

import torch

from hexo_utils.encoding import D6Symmetry, IDENTITY_D6


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


def augment_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    symmetry: D6Symmetry = IDENTITY_D6,
) -> dict[str, torch.Tensor]:
    """Apply the sampled symmetry to state, legal mask, and policy target.

    This remains a square-crop approximation until the encoder exposes exact
    axial D6 transforms. The important contract is that selection lives in
    `hexo_train`; the model receives and applies the chosen symmetry
    consistently.
    """

    transform = symmetry.index % 8
    augmented: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        augmented[key] = _transform_tensor(value, transform) if key in SPATIAL_KEYS else value
    return augmented
