"""Losses and value-bin helpers for dense CNN Model 1.

Model 1 predicts scalar value as a 65-bin distribution over `[-1, 1]`, not as a
single regression output. Policy-like targets are dense crop distributions. The
loss helpers validate target shape, finiteness, non-negativity, and probability
mass before normalizing so bad replay samples or evaluator bugs fail where the
training target is consumed.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import VALUE_BINS


def value_bins(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return the fixed 65 scalar support points for the value distribution."""

    return torch.linspace(-1.0, 1.0, VALUE_BINS, device=device, dtype=dtype)


def decode_binned_value(logits: torch.Tensor) -> torch.Tensor:
    """Decode value-bin logits to the expected scalar value."""

    bins = value_bins(device=logits.device, dtype=logits.dtype)
    return (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)


def scalar_to_binned_target(values: torch.Tensor | float) -> torch.Tensor:
    """Convert scalar values in `[-1, 1]` into adjacent-bin soft targets."""

    target = torch.as_tensor(values)
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("value targets must be finite")
    if bool(((target < -1.0) | (target > 1.0)).any().item()):
        raise ValueError("value targets must be in [-1, 1]")
    original_shape = target.shape
    flat = target.reshape(-1)
    position = (flat + 1.0) * ((VALUE_BINS - 1) / 2.0)
    lower = torch.floor(position).to(dtype=torch.long)
    upper = torch.ceil(position).to(dtype=torch.long)
    upper_weight = position - lower.to(dtype=position.dtype)
    lower_weight = 1.0 - upper_weight

    out = torch.zeros((flat.numel(), VALUE_BINS), device=flat.device, dtype=target.dtype)
    rows = torch.arange(flat.numel(), device=flat.device)
    out[rows, lower] += lower_weight
    out[rows, upper] += upper_weight
    return out.reshape(*original_shape, VALUE_BINS)


def soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_zero_rows: bool = False,
) -> torch.Tensor:
    """Cross entropy for dense policy distributions with strict target checks."""

    target = target.to(device=logits.device, dtype=logits.dtype)
    if logits.shape != target.shape:
        raise ValueError(f"cross entropy target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("cross entropy targets must be finite")
    if bool((target < 0).any().item()):
        raise ValueError("cross entropy targets must be nonnegative")
    row_sum = target.sum(dim=-1, keepdim=True)
    positive = row_sum > 0
    if not allow_zero_rows and not bool(positive.all().item()):
        raise ValueError("cross entropy targets must contain positive probability mass")
    normalizer = torch.where(positive, row_sum, torch.ones_like(row_sum))
    target = target / normalizer
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def binned_value_loss(
    logits: torch.Tensor,
    target: torch.Tensor | float,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy against scalar or distributional value targets."""

    target_tensor = torch.as_tensor(target, device=logits.device, dtype=logits.dtype)
    if target_tensor.shape != logits.shape:
        target_tensor = scalar_to_binned_target(target_tensor).to(device=logits.device, dtype=logits.dtype)
    if logits.shape != target_tensor.shape:
        raise ValueError(f"value target shape {tuple(target_tensor.shape)} does not match logits {tuple(logits.shape)}")
    if not bool(torch.isfinite(target_tensor).all().item()):
        raise ValueError("value distribution targets must be finite")
    if bool((target_tensor < 0).any().item()):
        raise ValueError("value distribution targets must be nonnegative")
    target_sum = target_tensor.sum(dim=-1, keepdim=True)
    if not bool((target_sum > 0).all().item()):
        raise ValueError("value distribution targets must contain positive probability mass")
    target_tensor = target_tensor / target_sum
    per_item = -(target_tensor * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if mask is None:
        return per_item.mean()
    mask_tensor = torch.as_tensor(mask, device=logits.device, dtype=logits.dtype)
    while mask_tensor.ndim < per_item.ndim:
        mask_tensor = mask_tensor.unsqueeze(-1)
    mask_tensor = mask_tensor.expand_as(per_item)
    denominator = mask_tensor.sum()
    if float(denominator.detach().cpu().item()) <= 0.0:
        return logits.sum() * 0.0
    return (per_item * mask_tensor).sum() / denominator


def model1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    opp_policy_weight: float = 0.25,
    lookahead_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted loss surface expected by `DenseCNNTrainer`."""

    components: dict[str, torch.Tensor] = {}
    components["policy"] = soft_cross_entropy(outputs["policy"], batch["policy"])
    components["value"] = binned_value_loss(outputs["value"], batch["value"])
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        components["opp_policy"] = soft_cross_entropy(outputs["opp_policy"], batch["opp_policy"], allow_zero_rows=True)
        total = total + opp_policy_weight * components["opp_policy"]

    for key, output in outputs.items():
        if key.startswith("lookahead_") and key in batch:
            components[key] = binned_value_loss(output, batch[key], mask=batch.get(f"{key}_mask"))
            total = total + lookahead_weight * components[key]

    components["total"] = total
    return total, components
