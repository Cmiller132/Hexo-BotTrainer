"""Coords-direct attention serve path (HEXFIELD_TRITON_ATTN2, _triton_attn.py
attn_coords): row-semantics parity against the model's pair build (CPU) and
kernel parity against the materialized reference (CUDA).

The kernel computes the bias row in-kernel from coords + the cell LUT; these
tests pin (1) that the reference reimplementation of the row build matches
model._build_pair_u8 exactly — the row semantics contract — and (2) that the
Triton kernel matches the materialized reference on live rows to the fp16
serve-parity class. Pad-QUERY rows are excluded from (2): the kernel stores
zeros where the reference computes downstream-masked garbage (both are
multiplied by seq_mask in the trunk)."""

from __future__ import annotations

import math

import pytest
import torch

from hexfield_eq import _triton_attn as TA
from hexfield_eq.constants import (
    BIAS_CELL_TOKEN_ROW,
    BIAS_ROWS,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    NUM_TOKENS,
)
from hexfield_eq.model import HexfieldNet


def _toy_geometry(b=3, n=40, seed=0):
    """Random-ish axial coords + a live mask with a padded tail and one
    interior dead cell (non-contiguous padding exercises the mask read)."""

    g = torch.Generator().manual_seed(seed)
    coords = torch.randint(-8, 9, (b, n, 2), generator=g, dtype=torch.int64)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, n - 6 :] = False  # pad tail
    mask[1, 7] = False  # interior dead cell (non-contiguous padding)
    return coords, mask


def test_ref_pair_build_matches_model_build_pair_u8():
    """_attn_coords_ref's internal row build == model._build_pair_u8 (the row
    semantics the kernel implements). Compared via the bias values the rows
    select, on live-query rows (pad-query rows are garbage on every path)."""

    torch.manual_seed(0)
    model = HexfieldNet()
    coords, mask = _toy_geometry()
    b, n, _ = coords.shape
    s = NUM_TOKENS + n
    pair_model = model._build_pair_u8(coords, mask)

    # Rebuild through the ref's code path: identical inputs, u8 LUT buffer.
    lut = model._cell_bias_lut_u8
    w = int(math.isqrt(int(lut.numel())))
    m = (w - 1) // 2
    c = coords.to(torch.int32)
    dq = c[:, None, :, 0] - c[:, :, None, 0]
    dr = c[:, None, :, 1] - c[:, :, None, 1]
    qi = (dq.clamp(-m, m) + m).to(torch.long)
    ri = (dr.clamp(-m, m) + m).to(torch.long)
    cell = lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)
    pair_ref = torch.full((b, s, s), BIAS_TOKEN_TOKEN_ROW, dtype=torch.uint8)
    pair_ref[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
    pair_ref[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
    pair_ref[:, NUM_TOKENS:, NUM_TOKENS:] = cell
    key_dead = torch.cat([mask.new_zeros(b, NUM_TOKENS), ~mask], dim=1)
    pair_ref = pair_ref.masked_fill(key_dead[:, None, :], BIAS_ROWS)

    assert torch.equal(pair_model, pair_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("d", [32, 64])
def test_attn_coords_kernel_matches_ref(d):
    if not TA.HAVE_TRITON:
        pytest.skip("no triton")
    torch.manual_seed(1)
    dev = torch.device("cuda")
    b, n, h = 3, 40, 3
    s = NUM_TOKENS + n
    coords, mask = _toy_geometry()
    coords = coords.to(dev)
    mask = mask.to(dev)
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.float16)
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.float16)
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.float16)
    table = torch.randn(BIAS_ROWS, h, dtype=torch.float16, device=dev) * 0.5
    pad_row = table.new_full((1, h), -3.0e4)
    table2 = torch.cat([table, pad_row], dim=0)
    # LUT like the model's: values < BIAS_ROWS - 3 (cell rows only).
    lut = torch.randint(0, 234, (35 * 35,), dtype=torch.uint8, device=dev)
    cell_idx = torch.arange(1, n + 1, device=dev, dtype=torch.int32)
    seq_lens = NUM_TOKENS + (cell_idx * mask).amax(dim=1)

    co_i32 = coords.to(torch.int32).contiguous()
    mask_u8 = mask.to(torch.uint8)
    out_k = TA.attn_coords(q, k, v, co_i32, mask_u8, table2, lut, seq_lens)
    if d in TA._ATTN2_FAILED:
        # Per-arch triton codegen edge case (see _triton_conv.py): the op fell
        # back to the reference, so there is no kernel to validate HERE — the
        # serve does the same fallback. The WSL GPU lane is the parity gate.
        pytest.skip(f"triton attn_coords compile failed for d={d} on this arch")
    out_r = TA._attn_coords_ref(q, k, v, co_i32, mask_u8, table2, lut, seq_lens)

    live_q = (
        torch.arange(s, device=dev)[None, :] < seq_lens[:, None]
    )[:, None, :, None]
    diff = ((out_k.float() - out_r.float()) * live_q).abs().max().item()
    assert diff < 2e-2, f"kernel vs ref masked max-abs {diff}"
