"""Losses and value-bin helpers for hexgt (Model 2).

Value is the same 65-bin distribution over `[-1, 1]` as dense_cnn (the binning
math is reused verbatim — pipeline-convenient and proven, §2.2/§6.4). The policy
differs: it is a *dynamic, per-candidate* distribution of variable length, packed
across a batch of graphs. So the policy/opp_policy loss is a **segmented**
softmax cross-entropy, normalized per graph via `candidate_graph` segment ids,
instead of dense_cnn's fixed `(N, 1681)` row CE.

All helpers validate target shape, finiteness, non-negativity, and probability
mass before normalizing so a bad replay sample or evaluator bug fails where the
training target is consumed.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import VALUE_BINS


# --- 65-bin value head (reused verbatim from dense_cnn) -----------------------

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


# --- Dynamic per-candidate (segmented) policy ---------------------------------

def segment_log_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Numerically-stable log-softmax over variable-length segments.

    `logits` is (Ctot,), `segment_ids` (Ctot,) maps each entry to its graph in
    `[0, num_segments)`. Returns log-softmax computed independently per segment.
    """

    if logits.ndim != 1:
        raise ValueError(f"segment_log_softmax expects 1-D logits, got shape {tuple(logits.shape)}")
    if segment_ids.shape != logits.shape:
        raise ValueError("segment_ids must match logits shape")
    seg = segment_ids.to(dtype=torch.long)
    seg_max = logits.new_full((num_segments,), float("-inf"))
    seg_max = seg_max.scatter_reduce(0, seg, logits, reduce="amax", include_self=True)
    # Segments with no entries keep -inf; they are never gathered below.
    shifted = logits - seg_max[seg]
    exp = shifted.exp()
    seg_sum = logits.new_zeros(num_segments).index_add(0, seg, exp)
    log_norm = seg_sum.clamp_min(torch.finfo(logits.dtype).tiny).log()
    return shifted - log_norm[seg]


def segment_softmax_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    *,
    allow_zero_segments: bool = False,
) -> torch.Tensor:
    """Per-graph cross entropy for the dynamic candidate policy.

    `target` is a (Ctot,) nonnegative per-candidate weight (e.g. visit counts).
    Each graph's target is normalized to a distribution over its candidates, then
    the mean cross entropy across graphs (with positive mass) is returned.
    """

    target = target.to(device=logits.device, dtype=logits.dtype)
    if target.shape != logits.shape:
        raise ValueError(f"policy target shape {tuple(target.shape)} does not match logits {tuple(logits.shape)}")
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("policy targets must be finite")
    if bool((target < 0).any().item()):
        raise ValueError("policy targets must be nonnegative")
    seg = segment_ids.to(dtype=torch.long)
    seg_mass = logits.new_zeros(num_segments).index_add(0, seg, target)
    positive = seg_mass > 0
    if not allow_zero_segments and not bool(positive.all().item()):
        raise ValueError("policy targets must contain positive probability mass per graph")
    # Normalize each graph's target to a distribution (guard zero-mass graphs).
    norm = torch.where(positive, seg_mass, torch.ones_like(seg_mass))
    target_normed = target / norm[seg]
    log_probs = segment_log_softmax(logits, seg, num_segments)
    per_candidate = -(target_normed * log_probs)
    per_segment = logits.new_zeros(num_segments).index_add(0, seg, per_candidate)
    denom = int(positive.sum().item())
    if denom == 0:
        return logits.sum() * 0.0
    return per_segment[positive].sum() / denom


def hexgt_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    opp_policy_weight: float = 0.25,
    short_term_value_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted loss surface expected by `HexgtTrainer`.

    `batch` carries `candidate_graph` (segment ids), `num_graphs`, the
    per-candidate `policy`/`opp_policy` targets, and the per-graph `value`/
    `stvalue_*` targets.
    """

    seg = batch["candidate_graph"]
    num_graphs = int(batch["num_graphs"])

    components: dict[str, torch.Tensor] = {}
    components["policy"] = segment_softmax_cross_entropy(
        outputs["policy"], batch["policy"], seg, num_graphs
    )
    components["value"] = binned_value_loss(outputs["value"], batch["value"])
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        components["opp_policy"] = segment_softmax_cross_entropy(
            outputs["opp_policy"], batch["opp_policy"], seg, num_graphs, allow_zero_segments=True
        )
        total = total + opp_policy_weight * components["opp_policy"]

    for key, output in outputs.items():
        if key.startswith("stvalue_") and key in batch:
            components[key] = binned_value_loss(output, batch[key], mask=batch.get(f"{key}_mask"))
            total = total + short_term_value_weight * components[key]

    components["total"] = total
    return total, components
