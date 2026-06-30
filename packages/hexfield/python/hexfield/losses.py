"""Losses and 65-bin helpers.

Two structural points:

- Policy CE is a segment soft cross-entropy over each row's legal prefix
  (scatter-logsumexp, fp32). No -1e9 fill exists in the loss path at all —
  legality masking is structural because the logit support IS the legal set.
  Target mass off the legal prefix is a hard error.
- Loss reduction is always mean over ROWS, never over nodes. Every reduction
  takes an optional explicit denominator so the pair-budget micro-bucket trainer
  can pass step-global denominators, making gradient accumulation exactly equal
  to a monolithic batch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F

from .constants import MOVES_LEFT_CAP, VALUE_BINS

POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
OPP_POLICY_WEIGHT = 0.25
# KataGo auxiliary SOFT policy target loss weight (main_4). KataGo's
# -soft-policy-weight-scale default is 8.0 (8x the main policy loss) paired with a
# T=4 (^0.25) full-legal target. We use a HEXO-ADAPTED gentler target (T=2 ^0.5,
# support-only — see batching.py), so we halve the weight to 4.0: with a sharper
# (less-flattened) soft target the per-row gradient is larger, so the 8x KataGo
# multiplier would over-pull the trunk. Mirror in config.TrainingSection
# (soft_policy_weight) — keep the two in sync.
SOFT_POLICY_WEIGHT = 4.0
SHORT_TERM_VALUE_WEIGHT = 0.1
MOVES_LEFT_WEIGHT = 0.1
Q_HEAD_WEIGHT = 0.1


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """fp32 floor for the loss math: upcast half/bfloat16, keep fp32/fp64.

    AMP-safe, not a downcast — the exactness tests run the whole loss path in
    fp64."""

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
    """Scalars in [-1, 1] -> adjacent-bin soft targets."""

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
    contribute exactly 0 but stay in the denominator (``allow_zero_rows``).
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
    if denom <= 0.0:
        # All-masked nominal batch (e.g. policy_rows==0): contribute exactly 0
        # without a divide-by-zero, mirroring binned_value_loss.
        return logits.sum() * 0.0
    return per_row.sum() / denom


def binned_value_loss(
    logits: torch.Tensor,
    target: torch.Tensor | float,
    *,
    mask: torch.Tensor | None = None,
    denominator: float | None = None,
) -> torch.Tensor:
    """CE against scalar or distributional 65-bin targets; masked rows
    contribute exactly 0; denominator defaults to the masked row
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
    soft_policy_weight: float = SOFT_POLICY_WEIGHT,
    short_term_value_weight: float = SHORT_TERM_VALUE_WEIGHT,
    moves_left_weight: float = MOVES_LEFT_WEIGHT,
    q_head_weight: float = Q_HEAD_WEIGHT,
    policy_target: str = "visit",
    denominators: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total = 1.0*policy + 1.0*value + 0.25*opp + 0.1*sum(stv) + 0.1*ml.

    ``denominators`` (step-global, computed at collate over the whole nominal
    batch) keys: ``rows`` plus per-masked-head row counts (``value``,
    ``stvalue_<h>``, ``moves_left``, ``cell_q``). When absent, this micro-bucket's
    own counts are used — correct for monolithic batches only. The ``value``
    count excludes truncated-game rows (value_mask==0) so they contribute zero to
    both numerator and denominator; with no truncated rows it equals ``rows``.
    """

    denoms = dict(denominators or {})
    rows = denoms.get("rows")
    components: dict[str, torch.Tensor] = {}

    # PCR value-rows: FAST rows carry all-zero policy + policy_valid==0. Fold the
    # policy_valid mask into the per-row CE weight so fast rows contribute exactly
    # 0 AND set allow_zero_rows so the zero-mass fast policy targets do not raise.
    # weight_denominator stays the FULL-row surprise-weight sum (G2), preserving
    # mean-over-full-rows. On full-only batches policy_valid is all-1 ⇒ the weight,
    # the denominator, and allow_zero_rows (no zero-mass rows present) all reduce
    # to the pre-fix path byte-identically.
    _pol_weight = batch.get("policy_ce_weight")
    _pv = batch.get("policy_valid")
    if _pol_weight is not None and _pv is not None:
        _pol_weight = _pol_weight * _pv
    # main_6 Gumbel S5: select the MAIN-policy CE target. With policy_target ==
    # "gumbel" AND a per-row gumbel target present (gumbel_policy_valid==1), drive
    # the CE from the improved-policy target π'; every other row (fast / absent /
    # old shard) keeps the visit target. Built as a per-row blend so a mixed batch
    # is correct and a visit-only batch (no gumbel cols, or policy_target=="visit")
    # is byte-identical to the pre-Gumbel path.
    _policy_target = batch["policy"]
    if (
        policy_target == "gumbel"
        and "gumbel_policy" in batch
        and "gumbel_policy_valid" in batch
    ):
        _use_gumbel = (batch["gumbel_policy_valid"] > 0.0).to(_policy_target.dtype)
        _use_gumbel = _use_gumbel.unsqueeze(1)  # (B,1) broadcast over the legal prefix
        _policy_target = (
            _use_gumbel * batch["gumbel_policy"] + (1.0 - _use_gumbel) * batch["policy"]
        )
    components["policy"] = segment_policy_ce(
        outputs["policy"],
        batch["legal_counts"],
        _policy_target,
        allow_zero_rows=True,
        row_weight=_pol_weight,
        weight_denominator=denoms.get("policy_ce_weight_sum"),
        denominator=rows,
    )
    # Value head masks truncated-game rows (value_mask==0): they carry no real
    # winner, so the hard-z target is undefined and must not pollute the head.
    # When no truncated rows are present, value_mask is all-1 and denoms['value']
    # == rows, so (per_item * 1).sum() / rows reproduces the unmasked path exactly
    # — completed-only batches are byte-identical. The denominator prefers the
    # step-global value count; absent that it falls back to `rows` (then to B
    # inside binned_value_loss), exactly as before. Callers that omit value_mask
    # entirely (mask=None) keep the original unmasked behavior byte-for-byte.
    components["value"] = binned_value_loss(
        outputs["value"],
        batch["value"],
        mask=batch.get("value_mask"),
        denominator=denoms.get("value", rows),
    )
    total = policy_weight * components["policy"] + value_weight * components["value"]

    if "opp_policy" in outputs and "opp_policy" in batch:
        # Zero-target rows (no future opponent decision / masked-from-fast / zero
        # projected mass) contribute exactly 0 but stay in the denominator
        # (allow_zero_rows). PCR value-rows BUGFIX (2026-06-22): FAST rows are NOT
        # guaranteed zero-target — a fast row whose NEXT move was a FULL move carries
        # a real future-opponent target. Without gating, those leaked gradient into
        # the opp head AND inflated the loss (numerator over all rows, denominator
        # policy_rows). row_weight=policy_valid zeroes fast rows in the numerator;
        # weight_denominator=policy_rows keeps the mean over FULL rows. On full-only
        # batches policy_valid is all-1 ⇒ byte-identical to pre-fix.
        components["opp_policy"] = segment_policy_ce(
            outputs["opp_policy"],
            batch["legal_counts"],
            batch["opp_policy"],
            allow_zero_rows=True,
            row_weight=batch.get("policy_valid"),
            weight_denominator=denoms.get("policy_rows", rows),
            denominator=denoms.get("policy_rows", rows),
        )
        total = total + opp_policy_weight * components["opp_policy"]

    if "soft_policy" in outputs and "soft_policy" in batch:
        # KataGo auxiliary SOFT policy: CE of the soft-policy head logits against
        # the (visit_policy+1e-7)^(1/4)-renormalized target (computed in
        # collate_training). FLAT rows denominator — KataGo's aux soft loss is not
        # surprise-reweighted (it carries no policy_ce_weight). The soft target is
        # derived from the main visit policy which always carries positive mass
        # (expand hard-errors on zero policy mass), so allow_zero_rows is not
        # needed.
        # FAST rows have all-zero soft_policy (derived from the all-zero visit
        # policy) → zero-mass rows. allow_zero_rows lets them contribute 0 without
        # raising; the denominator is gated to FULL rows so the mean is not
        # diluted. On full-only batches every soft row has positive mass and
        # policy_rows == rows ⇒ byte-identical to pre-fix (allow_zero_rows is a
        # no-op when no zero-mass row exists).
        components["soft_policy"] = segment_policy_ce(
            outputs["soft_policy"],
            batch["legal_counts"],
            batch["soft_policy"],
            allow_zero_rows=True,
            denominator=denoms.get("policy_rows", rows),
        )
        total = total + soft_policy_weight * components["soft_policy"]

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
