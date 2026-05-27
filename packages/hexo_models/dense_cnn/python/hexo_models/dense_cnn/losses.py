"""Losses and binned value decoding for Model 1."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import VALUE_BINS


def value_bins(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, VALUE_BINS, device=device, dtype=dtype)


def decode_binned_value(logits: torch.Tensor) -> torch.Tensor:
    bins = value_bins(device=logits.device, dtype=logits.dtype)
    return (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)


def scalar_to_binned_target(values: torch.Tensor | float) -> torch.Tensor:
    target = torch.as_tensor(values)
    target = target.clamp(-1.0, 1.0)
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


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def binned_value_loss(logits: torch.Tensor, target: torch.Tensor | float) -> torch.Tensor:
    target_tensor = torch.as_tensor(target, device=logits.device, dtype=logits.dtype)
    if target_tensor.shape != logits.shape:
        target_tensor = scalar_to_binned_target(target_tensor).to(device=logits.device, dtype=logits.dtype)
    return soft_cross_entropy(logits, target_tensor)


def model1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    opp_policy_weight: float = 0.25,
    lookahead_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components: dict[str, torch.Tensor] = {}
    components["policy"] = soft_cross_entropy(outputs["policy"], batch["policy"])
    components["value"] = binned_value_loss(outputs["value"], batch["value"])
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        components["opp_policy"] = soft_cross_entropy(outputs["opp_policy"], batch["opp_policy"])
        total = total + opp_policy_weight * components["opp_policy"]

    for key, output in outputs.items():
        if key.startswith("lookahead_") and key in batch:
            components[key] = binned_value_loss(output, batch[key])
            total = total + lookahead_weight * components[key]

    components["total"] = total
    return total, components
