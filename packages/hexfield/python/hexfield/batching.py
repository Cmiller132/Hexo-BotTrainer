"""Batch assembly for variable-N rows: the model collate, the training
collate, and the pair-budget micro-bucket split (spec §6.3).

Conventions consumed by `model.HexfieldNet` (see its docstring): pad rows are
all-zero features with nbr pointing at the appended zero row (index Npad) and
mask False; coords of pad rows are zero (never read — pad keys are additively
masked, pad query rows re-zeroed).

The pair budget `B_g * S_pad^2 <= PAIR_BUDGET` bounds the dominant
(B, heads, S, S) attention-bias transient (≈160 MB in fp16 under autocast).
S_pad here is the SAME quantity the model actually allocates: the trainer and
prefit pad every micro-bucket up to `ceil(maxN / PAD_QUANTUM) * PAD_QUANTUM`
nodes before appending NUM_TOKENS, so the split MUST quantize identically —
otherwise the live transient can exceed the budget by up to ~3.85x (the
worst case at N just above a 256 boundary). `pair_budget_microbuckets`
therefore takes the same quantum the padders use (default PAD_QUANTUM).
One optimizer step per nominal batch via gradient accumulation with
STEP-GLOBAL denominators (per-head unmasked-row counts computed here over the
whole nominal batch), which under LN is mathematically identical to a
monolithic batch — enforced by tests, not assumed.
"""

from __future__ import annotations

import numpy as np
import torch

from .constants import NUM_TOKENS
from .samples import ExpandedRow
from .support import Support

PAIR_BUDGET = 2.0e7
# Npad is quantized to multiples of this before NUM_TOKENS are appended (the
# §5.3 static-shape discipline: a small, repeating set of tensor shapes keeps
# the CUDA caching allocator from fragmenting toward the VRAM ceiling). The
# trainer and prefit pad to this same quantum; the budget split must too.
PAD_QUANTUM = 256


def quantized_npad(max_nodes: int, quantize: int = PAD_QUANTUM) -> int:
    """Round `max_nodes` up to a multiple of `quantize` (the padded Npad).

    `quantize <= 1` means no rounding (raw N) — used only by callers that
    deliberately collate without 256-quantization.
    """

    if quantize <= 1:
        return int(max_nodes)
    return -(-int(max_nodes) // quantize) * quantize


def collate_rows(
    rows: list[tuple[Support, np.ndarray]],
    pad_to: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a list of (Support, features) rows into one model batch."""

    npad = max(sup.num_nodes for sup, _ in rows)
    if pad_to is not None:
        if pad_to < npad:
            raise ValueError(f"pad_to {pad_to} < largest row {npad}")
        npad = pad_to
    b = len(rows)
    f = rows[0][1].shape[1]

    feats = torch.zeros(b, npad, f, dtype=torch.float32)
    nbr = torch.full((b, npad, 6), npad, dtype=torch.long)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    coords = torch.zeros(b, npad, 2, dtype=torch.long)
    legal_counts = torch.zeros(b, dtype=torch.long)

    for g, (sup, row_feats) in enumerate(rows):
        n = sup.num_nodes
        feats[g, :n] = torch.from_numpy(row_feats)
        row_nbr = torch.from_numpy(sup.nbr.astype(np.int64))
        nbr[g, :n] = torch.where(row_nbr >= 0, row_nbr, torch.full_like(row_nbr, npad))
        mask[g, :n] = True
        coords[g, :n] = torch.from_numpy(sup.coords.astype(np.int64))
        legal_counts[g] = sup.legal_count

    return {
        "feats": feats,
        "nbr": nbr,
        "mask": mask,
        "coords": coords,
        "legal_counts": legal_counts,
    }


def policy_surprise_weights(
    surprises: list[float], uniform_fraction: float, max_weight: float
) -> tuple[list[float], float]:
    """Per-row self-policy CE weights from KataGo policy-surprise (spec §5/§10.1).

    ``weight_g = uniform_fraction + (1 - uniform_fraction) * n * surprise_g / Σsurprise``
    clamped to ``max_weight``. Mean-1 by construction when no clamp fires (the
    normalization preserves mean-over-ROWS). All-zero surprise ⇒ every weight 1.0.
    Returns ``(weights, Σweights)``; the sum is the step-global self-CE denominator.
    """
    n = float(len(surprises))
    s = [max(0.0, float(x)) for x in surprises]
    total = sum(s)
    if total > 0.0:
        kl_frac = 1.0 - uniform_fraction
        w = [min(max_weight, uniform_fraction + kl_frac * n * x / total) for x in s]
    else:
        w = [1.0] * len(s)
    return w, float(sum(w))


def collate_training(
    rows: list[ExpandedRow],
    pad_to: int | None = None,
    *,
    row_weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Model batch + legal-prefix targets for one (micro-)batch of rows.

    ``row_weights`` (when given) is the precomputed per-row self-policy CE weight
    for THIS bucket's rows, sliced from the whole-nominal-batch weight vector by
    the caller (the weight is a function of the whole batch's surprise total, so
    it cannot be recomputed from the bucket alone — spec §10.4). ``None`` packs
    all-ones (legacy/standalone callers stay byte-identical).
    """

    batch = collate_rows([(row.support, row.feats) for row in rows], pad_to=pad_to)
    npad = batch["feats"].shape[1]
    b = len(rows)
    policy = torch.zeros(b, npad, dtype=torch.float32)
    opp = torch.zeros(b, npad, dtype=torch.float32)
    cell_q = torch.zeros(b, npad, dtype=torch.float32)
    cell_q_mask = torch.zeros(b, npad, dtype=torch.float32)
    # main_6 Gumbel S5: dense improved-policy target π' (B, L). All-zero for rows
    # without a target; gumbel_policy_valid flags which rows carry one so the loss
    # selects π' vs the visit target per row. Older ExpandedRow (no gumbel field)
    # ⇒ getattr fallback to all-zero / 0.0 (visit fallback) for transition safety.
    gumbel_policy = torch.zeros(b, npad, dtype=torch.float32)
    for g, row in enumerate(rows):
        n = row.policy.shape[0]
        policy[g, :n] = torch.from_numpy(row.policy)
        opp[g, :n] = torch.from_numpy(row.opp_policy)
        cell_q[g, :n] = torch.from_numpy(row.cell_q)  # n == row.cell_q.shape[0]
        cell_q_mask[g, :n] = torch.from_numpy(row.cell_q_mask)
        gp = getattr(row, "gumbel_policy", None)
        if gp is not None and gp.shape[0] == n:
            gumbel_policy[g, :n] = torch.from_numpy(gp)
    # KataGo auxiliary SOFT policy target (main_4), HEXO-ADAPTED (config review).
    # KataGo (metrics_pytorch.py) is target_soft = ((p + 1e-7)*policymask)^(1/T),
    # T=4 (exponent 0.25), over the FULL legal set. On Hexo that is harmful: the
    # legal prefix is 70-700+ cells, so the +1e-7 floor on every legal (incl.
    # UNVISITED) cell, amplified by ^0.25 ((1e-7)^0.25 ~= 0.018), leaks ~40-72% of
    # the soft-target mass onto the unvisited-legal tail — in sudden-death Hexo
    # that teaches a near-forced defensive block far too flat. Two adaptations:
    #   (1) SUPPORT-ONLY: transform ONLY the visited support (policy > 0); leave
    #       unvisited-legal cells at exactly 0 (no full-legal 1e-7 leak). Off-prefix
    #       slots also stay 0 (segment_policy_ce treats off-prefix mass as a hard
    #       error). The target is still a valid distribution over the support.
    #   (2) T=2 (exponent 0.5, gentler than KataGo's T=4) -> softens the visit
    #       distribution without the savage board-size-coupled flattening.
    # soft_policy_weight is correspondingly reduced 8.0 -> 4.0 in losses.py.
    # Pure function of `policy` (the packed visit policy) -> backend-agnostic: the
    # serial/pool/rust expand backends all produce the same `policy`, hence the
    # same soft target; no shard-schema or replay_expand.rs change is needed.
    soft_policy = torch.zeros(b, npad, dtype=torch.float32)
    legal_counts = batch["legal_counts"]
    prefix = (
        torch.arange(npad).unsqueeze(0) < legal_counts.unsqueeze(1)
    )  # (b, npad) bool, legal_counts == per-row n
    row_sum = policy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = policy / row_sum
    support = prefix & (policy > 0)  # visited support only (no full-legal 1e-7 leak)
    soft_prefix = p.pow(0.5)  # T=2 softening (gentler than KataGo's ^0.25)
    soft_policy[support] = soft_prefix[support]

    if row_weights is None:
        weights = [1.0] * b
    else:
        weights = list(row_weights)
    h = rows[0].stvalue.shape[0]
    batch.update(
        {
            "policy": policy,
            "soft_policy": soft_policy,
            "opp_policy": opp,
            "cell_q": cell_q,
            "cell_q_mask": cell_q_mask,
            # main_6 Gumbel S5: dense π' target + per-row presence mask. losses.py
            # drives the main-policy CE from gumbel_policy where valid (and
            # policy_target=="gumbel"); else the visit `policy`. All-zero / valid-0
            # on visit-only batches ⇒ byte-identical to pre-Gumbel.
            "gumbel_policy": gumbel_policy,
            "gumbel_policy_valid": torch.tensor(
                [float(getattr(row, "gumbel_policy_valid", 0.0)) for row in rows],
                dtype=torch.float32,
            ),
            "policy_ce_weight": torch.tensor(weights, dtype=torch.float32),
            "opp_coverage": torch.tensor([row.opp_coverage for row in rows]),
            "value": torch.tensor([row.value for row in rows], dtype=torch.float32),
            # Per-row value-head mask: 0.0 for truncated-game rows (no winner),
            # 1.0 for completed games. Gates the value loss to zero contribution
            # for truncated rows (parallel to moves_left_mask).
            "value_mask": torch.tensor(
                [row.value_mask for row in rows], dtype=torch.float32
            ),
            # Per-row policy-family mask: 0.0 for FAST (value-only) rows, 1.0 for
            # FULL rows. Gates self-policy CE, soft_policy CE, and opp_policy CE.
            # (cell_q is masked at expand via cell_q_mask.) All-1 on full-only
            # batches ⇒ byte-identical to pre-fix.
            "policy_valid": torch.tensor(
                [row.policy_valid for row in rows], dtype=torch.float32
            ),
            "stvalue": torch.stack(
                [torch.from_numpy(row.stvalue) for row in rows]
            ).reshape(b, h),
            "stvalue_mask": torch.stack(
                [torch.from_numpy(row.stvalue_mask) for row in rows]
            ).reshape(b, h),
            "moves_left": torch.tensor(
                [row.moves_left for row in rows], dtype=torch.float32
            ),
            "moves_left_mask": torch.tensor(
                [row.moves_left_mask for row in rows], dtype=torch.float32
            ),
        }
    )
    return batch


def split_stvalue_columns(
    batch: dict[str, torch.Tensor], horizons: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Per-horizon scalar targets/masks keyed the way `hexfield_loss` expects."""

    out = dict(batch)
    for col, horizon in enumerate(horizons):
        out[f"stvalue_{horizon}"] = batch["stvalue"][:, col]
        out[f"stvalue_{horizon}_mask"] = batch["stvalue_mask"][:, col]
    return out


def step_global_denominators(
    rows: list[ExpandedRow],
    horizons: tuple[int, ...],
    *,
    policy_surprise_uniform_fraction: float = 0.5,
    policy_surprise_max_weight: float = 8.0,
) -> dict[str, float]:
    """Per-head denominators over the WHOLE nominal batch (spec §6.3)."""

    denoms: dict[str, float] = {"rows": float(len(rows))}
    for col, horizon in enumerate(horizons):
        denoms[f"stvalue_{horizon}"] = float(
            sum(1.0 for row in rows if row.stvalue_mask[col] > 0)
        )
    # Value head: count only completed-game (unmasked) rows so truncated rows
    # neither contribute to the numerator nor inflate the denominator. With no
    # truncated rows this equals len(rows) (== the old `rows` denominator), so
    # completed-only batches are byte-identical.
    denoms["value"] = float(sum(row.value_mask for row in rows))
    denoms["moves_left"] = float(sum(row.moves_left_mask for row in rows))
    # cell_q: total masked-cell count (mean-over-contributing-cells, spec §10.3).
    denoms["cell_q"] = float(sum(float(row.cell_q_mask.sum()) for row in rows))
    # Flat full-row count: denominator for the opp/soft policy CE so the
    # mean-over-rows is not diluted ~3x by FAST (value-only) rows. On full-only
    # batches policy_rows == rows ⇒ byte-identical.
    denoms["policy_rows"] = float(sum(row.policy_valid for row in rows))
    # Self-policy CE weight sum over FULL rows only (fast rows are policy-masked;
    # including their surprise weight would inflate the mean-over-full-rows denom).
    # A FAST row has policy_surprise=0.0 → weight=uniform_fraction (not 0), so it
    # would WRONGLY inflate the denominator if counted; restrict to the full subset.
    full_surprises = [row.policy_surprise for row in rows if row.policy_valid != 0.0]
    _w, wsum = policy_surprise_weights(
        full_surprises,
        policy_surprise_uniform_fraction,
        policy_surprise_max_weight,
    )
    denoms["policy_ce_weight_sum"] = wsum
    return denoms


def pair_budget_microbuckets(
    rows: list[ExpandedRow],
    *,
    budget: float = PAIR_BUDGET,
    quantize: int = PAD_QUANTUM,
) -> list[list[ExpandedRow]]:
    """Sort a nominal batch by N and split under `B_g * S_pad^2 <= budget`.

    S_pad = ceil(largest N in the bucket / `quantize`) * `quantize` + NUM_TOKENS
    — the SAME padded sequence length the trainer/prefit actually allocate, so
    the live (B, heads, S_pad, S_pad) bias transient honours `budget`. (With
    `quantize <= 1` S_pad falls back to raw N + NUM_TOKENS.) A single
    over-budget row still forms its own bucket (it cannot be split further).
    """

    ordered = sorted(rows, key=lambda row: row.support.num_nodes)
    buckets: list[list[ExpandedRow]] = []
    current: list[ExpandedRow] = []
    for row in ordered:
        candidate = current + [row]
        # sorted ascending => candidate[-1] is the bucket's largest N
        npad = quantized_npad(candidate[-1].support.num_nodes, quantize)
        s_pad = npad + NUM_TOKENS
        if current and len(candidate) * (s_pad**2) > budget:
            buckets.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        buckets.append(current)
    return buckets
