"""Training losses for the Hexo ResNet plugin."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def policy_loss(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = policy_logits.flatten(start_dim=1)
    target = policy_target.flatten(start_dim=1).to(dtype=logits.dtype)
    if legal_mask is not None:
        mask = legal_mask.flatten(start_dim=1) > 0
        logits = logits.masked_fill(~mask, -1.0e9)
        target = target.masked_fill(~mask, 0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def value_loss(value: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(value.reshape_as(value_target), value_target)


def hexo_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> torch.Tensor:
    p_loss = policy_loss(
        outputs["policy_logits"],
        batch["policy_target"],
        batch.get("legal_mask"),
    )
    v_loss = value_loss(outputs["value"], batch["value_target"])
    return policy_weight * p_loss + value_weight * v_loss


def loss_components(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "policy_loss": policy_loss(
            outputs["policy_logits"],
            batch["policy_target"],
            batch.get("legal_mask"),
        ),
        "value_loss": value_loss(outputs["value"], batch["value_target"]),
    }

