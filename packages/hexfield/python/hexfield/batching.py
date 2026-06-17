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


def collate_training(
    rows: list[ExpandedRow], pad_to: int | None = None
) -> dict[str, torch.Tensor]:
    """Model batch + legal-prefix targets for one (micro-)batch of rows."""

    batch = collate_rows([(row.support, row.feats) for row in rows], pad_to=pad_to)
    npad = batch["feats"].shape[1]
    b = len(rows)
    policy = torch.zeros(b, npad, dtype=torch.float32)
    opp = torch.zeros(b, npad, dtype=torch.float32)
    for g, row in enumerate(rows):
        n = row.policy.shape[0]
        policy[g, :n] = torch.from_numpy(row.policy)
        opp[g, :n] = torch.from_numpy(row.opp_policy)
    h = rows[0].stvalue.shape[0]
    batch.update(
        {
            "policy": policy,
            "opp_policy": opp,
            "opp_coverage": torch.tensor([row.opp_coverage for row in rows]),
            "value": torch.tensor([row.value for row in rows], dtype=torch.float32),
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
    rows: list[ExpandedRow], horizons: tuple[int, ...]
) -> dict[str, float]:
    """Per-head denominators over the WHOLE nominal batch (spec §6.3)."""

    denoms: dict[str, float] = {"rows": float(len(rows))}
    for col, horizon in enumerate(horizons):
        denoms[f"stvalue_{horizon}"] = float(
            sum(1.0 for row in rows if row.stvalue_mask[col] > 0)
        )
    denoms["moves_left"] = float(sum(row.moves_left_mask for row in rows))
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
