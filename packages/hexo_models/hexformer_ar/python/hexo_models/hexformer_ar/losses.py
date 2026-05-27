"""Losses for sparse Hexformer AR training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def masked_soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    target = target.to(device=logits.device, dtype=logits.dtype) * mask.to(device=logits.device, dtype=logits.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return -(target * F.log_softmax(masked, dim=-1)).sum(dim=-1).mean()


def wdl_value_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs[..., 2] - probs[..., 0]


def policy_symmetry_consistency_loss(
    transformed_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL for two policy rows with aligned candidate order."""

    valid = mask.bool().to(device=transformed_logits.device)
    left = transformed_logits.masked_fill(~valid, torch.finfo(transformed_logits.dtype).min)
    right = reference_logits.to(device=left.device, dtype=left.dtype).masked_fill(~valid, torch.finfo(left.dtype).min)
    left_log = F.log_softmax(left, dim=-1)
    right_log = F.log_softmax(right, dim=-1)
    left_prob = left_log.exp()
    right_prob = right_log.exp()
    left_to_right = (left_prob * (left_log - right_log)).masked_fill(~valid, 0.0).sum(dim=-1)
    right_to_left = (right_prob * (right_log - left_log)).masked_fill(~valid, 0.0).sum(dim=-1)
    return 0.5 * (left_to_right + right_to_left).mean()


def hexformer_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = 1.0,
    wdl_weight: float = 1.0,
    distance_weight: float = 0.25,
    opponent_policy_weight: float = 0.15,
    lookahead_weight: float = 0.25,
    threat_weight: float = 0.5,
    relevance_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = batch["candidate_mask"].bool().to(device=outputs["policy_logits"].device)
    components: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=outputs["policy_logits"].device, dtype=outputs["policy_logits"].dtype)
    if "policy_target" in batch:
        components["policy"] = masked_soft_cross_entropy(outputs["policy_logits"], batch["policy_target"].to(mask.device), mask)
        total = total + policy_weight * components["policy"]
    if "wdl_target" in batch:
        target = batch["wdl_target"].to(device=outputs["wdl_logits"].device, dtype=outputs["wdl_logits"].dtype)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        components["wdl"] = -(target * F.log_softmax(outputs["wdl_logits"], dim=-1)).sum(dim=-1).mean()
        total = total + wdl_weight * components["wdl"]
    if "distance_target" in batch:
        target = batch["distance_target"].to(device=outputs["distance"].device, dtype=outputs["distance"].dtype)
        components["distance"] = F.smooth_l1_loss(outputs["distance"], target)
        total = total + distance_weight * components["distance"]
    if "opp_policy_target" in batch:
        components["opp_policy"] = masked_soft_cross_entropy(
            outputs["opp_policy_logits"],
            batch["opp_policy_target"].to(mask.device),
            mask,
        )
        total = total + opponent_policy_weight * components["opp_policy"]
    if "threat_target" in batch:
        threat_target = batch["threat_target"].to(device=outputs["threat_logits"].device, dtype=torch.long)
        threat_logits = outputs["threat_logits"].reshape(-1, outputs["threat_logits"].shape[-1])
        threat_mask = mask.reshape(-1)
        if bool(threat_mask.any()):
            components["threat"] = F.cross_entropy(threat_logits[threat_mask], threat_target.reshape(-1)[threat_mask])
            total = total + threat_weight * components["threat"]
    if "relevance_target" in batch:
        relevance = batch["relevance_target"].to(device=outputs["rz_logits"].device, dtype=outputs["rz_logits"].dtype)
        components["relevance"] = F.binary_cross_entropy_with_logits(outputs["rz_logits"][mask], relevance[mask])
        total = total + relevance_weight * components["relevance"]
    for key, output in outputs.items():
        if not key.startswith("lookahead_") or f"{key}_target" not in batch:
            continue
        target = batch[f"{key}_target"].to(device=output.device, dtype=output.dtype)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        components[key] = -(target * F.log_softmax(output, dim=-1)).sum(dim=-1).mean()
        total = total + lookahead_weight * components[key]
    components["total"] = total
    return total, components
