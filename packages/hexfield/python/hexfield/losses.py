"""Losses and 65-bin helpers — spec §3.

Target semantics are exact ports of the verified restnet constructions
(restnet losses.py is imported in tests as the oracle), re-keyed from crop
flats to support nodes. Two structural differences from the crop lineage:

- Policy CE is a segment soft cross-entropy over each row's legal prefix
  (scatter-logsumexp, fp32). No -1e9 fill exists in the loss path at all —
  legality masking is structural because the logit support IS the legal set.
  Target mass off the legal prefix is a hard error.
- Loss reduction is always mean over ROWS, never over nodes (the variable-N
  bug rule). Every reduction takes an optional explicit denominator so the
  pair-budget micro-bucket trainer (§6.3) can pass step-global denominators,
  making gradient accumulation exactly equal to a monolithic batch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import MOVES_LEFT_CAP, VALUE_BINS

POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
OPP_POLICY_WEIGHT = 0.25
SHORT_TERM_VALUE_WEIGHT = 0.1
MOVES_LEFT_WEIGHT = 0.1
Q_HEAD_WEIGHT = 0.1


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """fp32 floor for the loss math: upcast half/bfloat16, keep fp32/fp64.

    "fp32" in the spec means AMP-safe, not a downcast — the §6.3 exactness
    tests run the whole loss path in fp64."""

    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def value_bins(*, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """The fixed 65 scalar support points for every binned head."""

    return torch.linspace(-1.0, 1.0, VALUE_BINS, device=device, dtype=dtype)


def decode_binned_value(logits: torch.Tensor) -> torch.Tensor:
    """Softmax expectation, clamped to [-1, 1] (the serve-side decode)."""

    bins = value_bins(device=logits.device, dtype=logits.dtype)
    return ((torch.softmax(logits, dim=-1) * bins).sum(dim=-1)).clamp(-1.0, 1.0)


def decode_moves_left(logits: torch.Tensor) -> torch.Tensor:
    """Softmax-EXPECTATION decode mapped to decisions [0, MOVES_LEFT_CAP] (v3).

    Replaces the median-of-bins decode, whose ~8-decision quantization drove the
    full-horizon Spearman drift; expectation mirrors ``decode_binned_value``."""

    bins = value_bins(device=logits.device, dtype=logits.dtype)
    scalar = (torch.softmax(logits, dim=-1) * bins).sum(dim=-1).clamp(-1.0, 1.0)
    return (scalar + 1.0) * 0.5 * MOVES_LEFT_CAP


def scalar_to_binned_target(values: torch.Tensor | float) -> torch.Tensor:
    """Scalars in [-1, 1] -> adjacent-bin soft targets (restnet semantics)."""

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


def segment_policy_ce(
    logits: torch.Tensor,
    legal_counts: torch.Tensor,
    target: torch.Tensor,
    *,
    allow_zero_rows: bool = False,
    denominator: float | None = None,
    row_weight: torch.Tensor | None = None,
    weight_denominator: float | None = None,
) -> torch.Tensor:
    """Soft CE over each row's legal prefix; mean over rows.

    logits/target: (B, Npad); per row g only slots [0, L_g) participate.
    Target mass outside the prefix is a hard error (for opp-policy targets the
    projection/drop happens at expansion, never here). Zero-mass rows
    contribute exactly 0 but stay in the denominator (restnet
    ``allow_zero_rows`` semantics).
    """

    if logits.shape != target.shape:
        raise ValueError(
            f"policy target shape {tuple(target.shape)} != logits {tuple(logits.shape)}"
        )
    b, npad = logits.shape
    if bool((legal_counts <= 0).any().item()):
        raise ValueError("policy rows must have at least one legal move")
    prefix = torch.arange(npad, device=logits.device).unsqueeze(0) < legal_counts.unsqueeze(1)
    target = target.to(device=logits.device)
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("policy targets must be finite")
    if bool((target < 0).any().item()):
        raise ValueError("policy targets must be nonnegative")
    if bool((target[~prefix] > 0).any().item()):
        raise ValueError("policy target mass off the legal prefix is a hard error")

    row_sum = target.sum(dim=-1)
    positive = row_sum > 0
    if not allow_zero_rows and not bool(positive.all().item()):
        raise ValueError("policy targets must contain positive probability mass")

    flat_logits = _at_least_fp32(logits[prefix])
    flat_target = _at_least_fp32(target[prefix])
    row_ids = prefix.nonzero(as_tuple=True)[0]

    # Scatter-logsumexp per row (fp32): max, then sum of shifted exps.
    row_max = torch.full((b,), float("-inf"), device=logits.device, dtype=flat_logits.dtype)
    row_max = row_max.scatter_reduce(0, row_ids, flat_logits, reduce="amax")
    shifted = (flat_logits - row_max[row_ids]).exp()
    row_expsum = torch.zeros(b, device=logits.device, dtype=flat_logits.dtype)
    row_expsum = row_expsum.index_add(0, row_ids, shifted)
    lse = row_max + row_expsum.log()

    log_probs = flat_logits - lse[row_ids]
    normalizer = _at_least_fp32(torch.where(positive, row_sum, torch.ones_like(row_sum)))
    weighted = flat_target / normalizer[row_ids] * log_probs
    per_row = torch.zeros(b, device=logits.device, dtype=flat_logits.dtype)
    per_row = per_row.index_add(0, row_ids, weighted).neg()

    if row_weight is not None:
        per_row = per_row * row_weight.to(device=per_row.device, dtype=per_row.dtype)
        denom = float(b) if weight_denominator is None else float(weight_denominator)
    else:
        denom = float(b) if denominator is None else float(denominator)
    return per_row.sum() / denom


def binned_value_loss(
    logits: torch.Tensor,
    target: torch.Tensor | float,
    *,
    mask: torch.Tensor | None = None,
    denominator: float | None = None,
) -> torch.Tensor:
    """CE against scalar or distributional 65-bin targets (restnet semantics);
    masked rows contribute exactly 0; denominator defaults to the masked row
    count (or B when unmasked), overridable for step-global accumulation."""

    target_tensor = torch.as_tensor(target, device=logits.device, dtype=logits.dtype)
    if target_tensor.shape != logits.shape:
        target_tensor = scalar_to_binned_target(target_tensor).to(
            device=logits.device, dtype=logits.dtype
        )
    if logits.shape != target_tensor.shape:
        raise ValueError(
            f"value target shape {tuple(target_tensor.shape)} != logits {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(target_tensor).all().item()):
        raise ValueError("value distribution targets must be finite")
    if bool((target_tensor < 0).any().item()):
        raise ValueError("value distribution targets must be nonnegative")
    target_sum = target_tensor.sum(dim=-1, keepdim=True)
    if not bool((target_sum > 0).all().item()):
        raise ValueError("value distribution targets must contain positive probability mass")
    target_tensor = target_tensor / target_sum
    per_item = -(target_tensor * F.log_softmax(_at_least_fp32(logits), dim=-1)).sum(dim=-1)
    if mask is None:
        denom = float(per_item.numel()) if denominator is None else float(denominator)
        return per_item.sum() / denom
    mask_tensor = torch.as_tensor(mask, device=logits.device, dtype=per_item.dtype)
    while mask_tensor.ndim < per_item.ndim:
        mask_tensor = mask_tensor.unsqueeze(-1)
    mask_tensor = mask_tensor.expand_as(per_item)
    denom = float(mask_tensor.sum().item()) if denominator is None else float(denominator)
    if denom <= 0.0:
        return logits.sum() * 0.0
    return (per_item * mask_tensor).sum() / denom


def hexfield_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    policy_weight: float = POLICY_WEIGHT,
    value_weight: float = VALUE_WEIGHT,
    opp_policy_weight: float = OPP_POLICY_WEIGHT,
    short_term_value_weight: float = SHORT_TERM_VALUE_WEIGHT,
    moves_left_weight: float = MOVES_LEFT_WEIGHT,
    q_head_weight: float = Q_HEAD_WEIGHT,
    denominators: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total = 1.0*policy + 1.0*value + 0.25*opp + 0.1*sum(stv) + 0.1*ml.

    ``denominators`` (step-global, computed at collate over the whole nominal
    batch) keys: ``rows`` plus per-masked-head row counts (``stvalue_<h>``,
    ``moves_left``). When absent, this micro-bucket's own counts are used —
    correct for monolithic batches only.
    """

    denoms = dict(denominators or {})
    rows = denoms.get("rows")
    components: dict[str, torch.Tensor] = {}

    components["policy"] = segment_policy_ce(
        outputs["policy"],
        batch["legal_counts"],
        batch["policy"],
        row_weight=batch.get("policy_ce_weight"),
        weight_denominator=denoms.get("policy_ce_weight_sum"),
        denominator=rows,
    )
    components["value"] = binned_value_loss(
        outputs["value"], batch["value"], denominator=rows
    )
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        # Zero-target rows (no future opponent decision / masked-from-fast /
        # zero projected mass) contribute exactly 0 but stay in the
        # denominator — restnet allow_zero_rows semantics.
        components["opp_policy"] = segment_policy_ce(
            outputs["opp_policy"],
            batch["legal_counts"],
            batch["opp_policy"],
            allow_zero_rows=True,
            denominator=rows,
        )
        total = total + opp_policy_weight * components["opp_policy"]

    for key, output in outputs.items():
        if key.startswith("stvalue_") and key in batch:
            components[key] = binned_value_loss(
                output,
                batch[key],
                mask=batch.get(f"{key}_mask"),
                denominator=denoms.get(key),
            )
            total = total + short_term_value_weight * components[key]

    if "moves_left" in outputs and "moves_left" in batch:
        components["moves_left"] = binned_value_loss(
            outputs["moves_left"],
            batch["moves_left"],
            mask=batch.get("moves_left_mask"),
            denominator=denoms.get("moves_left"),
        )
        total = total + moves_left_weight * components["moves_left"]

    if "cell_q" in outputs and "cell_q" in batch:
        components["cell_q"] = binned_value_loss(
            outputs["cell_q"],
            batch["cell_q"],
            mask=batch["cell_q_mask"],
            denominator=denoms.get("cell_q"),
        )
        total = total + q_head_weight * components["cell_q"]

    components["total"] = total
    return total, components
