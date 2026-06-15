"""hexflash — shape-generic attention kernels for the variable-S rel-pos-bias
trunk attention (Layer A of the inference rewrite).

Two public backends, identical signature, both numerically equivalent (within
fp16 rounding) to the deployed `RelPosAttention` "materialized"/"sdpa" path:

  * ``hexflash_attention``        — hand-written fused FlashAttention-2 in Triton
                                    that reconstructs the rel-pos-bias ROW in
                                    kernel from the model's own ``_exact_lut`` +
                                    ``bias_table``. ONE binary covers Npad
                                    64..3300+ (S is a runtime arg). PRIMARY.
  * ``flex_attention_relpos``     — ``torch.nn.attention.flex_attention`` with a
                                    ``score_mod`` that replays the same row
                                    selection + a ``mask_mod`` for pad keys.
                                    FALLBACK.

Both reuse the bias-row INDEX math byte-for-byte from ``build_attn_bias``
(model.py) and the geometry LUT, so the bias VALUE per (query,key) pair is
bit-identical to the production path by construction (Tier-1 oracle). Only the
score+softmax+@V core changes kernel.

NO imports from model.py / inference.py — pure functions over tensors +
constants, so the parity oracle can import this without circularity.

Exactness invariants reproduced here (the basis for "no retrain"):
  * scale = 1/sqrt(HEAD_DIM=24) applied to q@kᵀ BEFORE the bias add.
  * PAD_KEY_MASK_VALUE = -3.0e4 (finite in fp16) added to pad KEY columns;
    token keys (slots < NUM_TOKENS) are NEVER masked.
  * head_dim 24 is run on a 32-wide tile with the last 8 lanes zero-loaded
    (zeros add 0 to every score; the extra V lanes are never written out).
  * Pad QUERY rows may compute garbage — that is correct, AttnBlock re-zeros
    them with ``*m`` downstream, so the row stays output-bit-identical.

The bias-row selection (must match model.py:274-296 exactly):
  d = max(|dq|, |dr|, |dq+dr|),  dq,dr = coords[key] - coords[query]
  d <= 8 (BIAS_DISK_RADIUS): exact_lut[(clamp(dq)+8)*17 + (clamp(dr)+8)]
  9 <= d <= 16 (BIAS_RING_MAX): (on-axis ? BIAS_ON_AXIS_BASE : BIAS_OFF_AXIS_BASE) + (d-9)
                                on-axis == dq==0 | dr==0 | dq+dr==0
  d >= 17: BIAS_FAR_ROW
  any slot < NUM_TOKENS: token-class rows
     token(q)–token(k): BIAS_TOKEN_TOKEN_ROW
     token(q)–cell(k) : BIAS_TOKEN_CELL_ROW
     cell(q)–token(k) : BIAS_CELL_TOKEN_ROW
"""

from __future__ import annotations

import math

import torch

from .constants import (
    ATTENTION_HEADS,
    BIAS_CELL_TOKEN_ROW,
    BIAS_DISK_RADIUS,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    HEAD_DIM,
    NUM_TOKENS,
)

# Imported from model.py would be circular; the value is part of the FROZEN
# contract (model.py:55) and is asserted equal in constants-owner's test.
PAD_KEY_MASK_VALUE = -3.0e4

try:  # Triton is optional at import time (CPU dev boxes have no CUDA Triton).
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


# =============================================================================
#  Shared host-side bias-row index builder (the Tier-1 oracle ground truth).
#  This is the SAME integer selection as model.build_attn_bias:274-296 but kept
#  here as a standalone pure function so it can be (a) used by the reference
#  backend, (b) imported by the parity test, and (c) cross-checked against
#  build_attn_bias without importing model.
# =============================================================================

def relpos_pair_index(
    coords: torch.Tensor,  # (B, Npad, 2) int (q, r); pad coords arbitrary
    exact_lut: torch.Tensor,  # (289,) long  == model._exact_lut
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:
    """(B, S, S) long bias-table ROW index, S = num_tokens + Npad.

    Bit-identical to the integer index computed inside ``build_attn_bias``
    (cell/cell block) plus the three token-class blocks. No masking, no gather
    of the table itself — just the row indices. ``[query, key]`` order, i.e.
    offset = coords[key] - coords[query] (matches model.py:272-273)."""

    b, n, _ = coords.shape
    cq = coords[..., 0].to(torch.long)
    cr = coords[..., 1].to(torch.long)
    dq = cq[:, None, :] - cq[:, :, None]  # (B, N, N) key - query
    dr = cr[:, None, :] - cr[:, :, None]
    d = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())

    R = BIAS_DISK_RADIUS
    clamped_q = dq.clamp(-R, R) + R
    clamped_r = dr.clamp(-R, R) + R
    exact = exact_lut[(clamped_q * 17 + clamped_r).reshape(-1)].reshape(b, n, n)

    on_axis = (dq == 0) | (dr == 0) | (dq + dr == 0)
    ring_base = torch.where(
        on_axis,
        torch.full_like(d, BIAS_ON_AXIS_BASE),
        torch.full_like(d, BIAS_OFF_AXIS_BASE),
    )
    ring = ring_base + (d - BIAS_RING_MIN)
    cell_idx = torch.where(
        d <= R,
        exact,
        torch.where(d <= BIAS_RING_MAX, ring, torch.full_like(d, BIAS_FAR_ROW)),
    )

    s = num_tokens + n
    pair = coords.new_full((b, s, s), BIAS_TOKEN_TOKEN_ROW, dtype=torch.long)
    pair[:, :num_tokens, num_tokens:] = BIAS_TOKEN_CELL_ROW
    pair[:, num_tokens:, :num_tokens] = BIAS_CELL_TOKEN_ROW
    pair[:, num_tokens:, num_tokens:] = cell_idx
    return pair


# =============================================================================
#  Reference backend: pure PyTorch, no Triton. Materialized-equivalent.
#  Used as a CPU/oracle path and whenever Triton is unavailable. This is the
#  semantic definition both fused kernels must reproduce.
# =============================================================================

def reference_relpos_attention(
    q: torch.Tensor,  # (B, H, S, Dh)
    k: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,  # (B, Npad, 2) int
    bias_table: torch.Tensor,  # (BIAS_ROWS, H)
    seq_mask: torch.Tensor,  # (B, S) bool, True = live key
    exact_lut: torch.Tensor,  # (289,) long
    scale: float,
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:
    """Materialized attention = the canonical math the fused kernels target.

    Identical to RelPosAttention.impl=='materialized' but builds the bias from
    (coords, table) instead of receiving a prebuilt (B,H,S,S)."""

    pair = relpos_pair_index(coords, exact_lut, num_tokens)  # (B, S, S)
    # (B, S, S, H) -> (B, H, S, S)
    bias = bias_table.to(q.dtype)[pair].permute(0, 3, 1, 2)
    fill = torch.where(seq_mask, 0.0, PAD_KEY_MASK_VALUE).to(q.dtype)
    bias = bias + fill[:, None, None, :]  # broadcast over (H, query)
    scores = (q @ k.transpose(-2, -1)) * scale + bias
    return torch.softmax(scores, dim=-1) @ v


# =============================================================================
#  Triton fused FlashAttention-2 with in-kernel rel-pos-bias reconstruction.
# =============================================================================

if _HAS_TRITON:

    @triton.jit
    def _hexflash_kernel(
        Q, K, V, Out,
        Coords,            # (B, Npad, 2) int32
        BiasTable,         # (BIAS_ROWS, H) fp32
        SeqMask,           # (B, S) int8 (1 = live key)
        ExactLut,          # (289,) int32
        scale,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_cb, stride_cn,           # Coords: batch, node (last dim contiguous=2)
        stride_btr, stride_bth,         # BiasTable: row, head
        stride_mb, stride_ms,           # SeqMask: batch, seq
        H: tl.constexpr,
        S,                              # runtime: NUM_TOKENS + Npad
        NPAD,                           # runtime: Npad
        NUM_TOKENS: tl.constexpr,
        DISK_RADIUS: tl.constexpr,
        RING_MAX: tl.constexpr,
        RING_MIN: tl.constexpr,
        ON_AXIS_BASE: tl.constexpr,
        OFF_AXIS_BASE: tl.constexpr,
        FAR_ROW: tl.constexpr,
        TOK_TOK_ROW: tl.constexpr,
        TOK_CELL_ROW: tl.constexpr,
        CELL_TOK_ROW: tl.constexpr,
        PAD_MASK_VALUE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,          # 32 (>= HEAD_DIM=24, padded lanes zeroed)
        HEAD_DIM: tl.constexpr,         # 24
    ):
        pid_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        b = off_bh // H
        h = off_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query rows
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM

        # --- load Q block (BLOCK_M, BLOCK_D), zero the pad lanes ---
        q_ptrs = (
            Q + b * stride_qb + h * stride_qh
            + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        )
        q_row_mask = offs_m[:, None] < S
        q = tl.load(q_ptrs, mask=q_row_mask & d_mask[None, :], other=0.0)
        q = q.to(tl.float32) * scale

        # --- query geometry (axial coords), token flag ---
        q_is_tok = offs_m < NUM_TOKENS
        q_cell = offs_m - NUM_TOKENS                       # cell index (valid if >=0)
        q_cell_clamped = tl.where(q_cell < 0, 0, q_cell)
        qc_ptr = Coords + b * stride_cb + q_cell_clamped * stride_cn
        q_qcoord = tl.load(qc_ptr + 0, mask=~q_is_tok & (offs_m < S), other=0).to(tl.int32)
        q_rcoord = tl.load(qc_ptr + 1, mask=~q_is_tok & (offs_m < S), other=0).to(tl.int32)

        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)       # key columns
            n_valid = offs_n < S

            k_ptrs = (
                K + b * stride_kb + h * stride_kh
                + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=n_valid[:, None] & d_mask[None, :], other=0.0)
            k = k.to(tl.float32)
            # scores (BLOCK_M, BLOCK_N) = (q*scale) @ kᵀ
            scores = tl.dot(q, tl.trans(k))

            # --- key geometry ---
            k_is_tok = offs_n < NUM_TOKENS
            k_cell = offs_n - NUM_TOKENS
            k_cell_clamped = tl.where(k_cell < 0, 0, k_cell)
            kc_ptr = Coords + b * stride_cb + k_cell_clamped * stride_cn
            k_qcoord = tl.load(kc_ptr + 0, mask=~k_is_tok & n_valid, other=0).to(tl.int32)
            k_rcoord = tl.load(kc_ptr + 1, mask=~k_is_tok & n_valid, other=0).to(tl.int32)

            # --- per-pair bias ROW index (BLOCK_M, BLOCK_N) ---
            dq = k_qcoord[None, :] - q_qcoord[:, None]
            dr = k_rcoord[None, :] - q_rcoord[:, None]
            adq = tl.abs(dq)
            adr = tl.abs(dr)
            adqr = tl.abs(dq + dr)
            dd = tl.maximum(tl.maximum(adq, adr), adqr)

            cq = tl.minimum(tl.maximum(dq, -DISK_RADIUS), DISK_RADIUS) + DISK_RADIUS
            cr = tl.minimum(tl.maximum(dr, -DISK_RADIUS), DISK_RADIUS) + DISK_RADIUS
            lut_idx = cq * 17 + cr
            exact_row = tl.load(ExactLut + lut_idx).to(tl.int32)

            on_axis = (dq == 0) | (dr == 0) | ((dq + dr) == 0)
            ring_base = tl.where(on_axis, ON_AXIS_BASE, OFF_AXIS_BASE)
            ring_row = ring_base + (dd - RING_MIN)
            cell_row = tl.where(
                dd <= DISK_RADIUS,
                exact_row,
                tl.where(dd <= RING_MAX, ring_row, FAR_ROW),
            )

            # token-class override (any slot < NUM_TOKENS)
            qtok = q_is_tok[:, None]
            ktok = k_is_tok[None, :]
            row = tl.where(
                qtok & ktok, TOK_TOK_ROW,
                tl.where(
                    qtok & ~ktok, TOK_CELL_ROW,
                    tl.where(~qtok & ktok, CELL_TOK_ROW, cell_row),
                ),
            )
            row = row.to(tl.int32)

            bias = tl.load(BiasTable + row * stride_btr + h * stride_bth).to(tl.float32)
            scores = scores + bias

            # --- pad KEY mask (token keys never masked) ---
            live = tl.load(
                SeqMask + b * stride_mb + offs_n * stride_ms,
                mask=n_valid, other=0,
            ).to(tl.int1)
            keep = n_valid & live
            scores = tl.where(keep[None, :], scores, PAD_MASK_VALUE)

            # --- online softmax (FA2) ---
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = (
                V + b * stride_vb + h * stride_vh
                + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=n_valid[:, None] & d_mask[None, :], other=0.0)
            acc += tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = (
            Out + b * stride_ob + h * stride_oh
            + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
                 mask=(offs_m[:, None] < S) & d_mask[None, :])


def _select_block(npad: int) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, num_warps) by S-bucket. Still ONE binary — this is
    launch-config selection (num_stages/num_warps/tile), not specialization on
    a constexpr shape. Conservative defaults; the GPU pause autotunes these."""
    if npad <= 256:
        return 64, 64, 4
    if npad <= 1024:
        return 128, 64, 4
    return 128, 128, 8


def hexflash_attention(
    q: torch.Tensor,       # (B, H, S, Dh=24) fp16/bf16
    k: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,  # (B, Npad, 2) int32
    bias_table: torch.Tensor,  # (BIAS_ROWS, H)
    seq_mask: torch.Tensor,    # (B, S) bool, True = live key
    exact_lut: torch.Tensor,   # (289,) int / long
    scale: float,
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:
    """Fused FA2 with in-kernel rel-pos bias. Returns (B, H, S, Dh).

    FROZEN public surface (model.py A2 + tests depend on this). On non-CUDA or
    when Triton is missing, falls back to the reference path so the function is
    always callable (the oracle exercises both)."""

    b, h, s, dh = q.shape
    assert dh == HEAD_DIM, f"hexflash expects head_dim={HEAD_DIM}, got {dh}"
    npad = coords.shape[1]
    assert s == num_tokens + npad, f"S={s} != num_tokens+Npad={num_tokens+npad}"

    if (not _HAS_TRITON) or (not q.is_cuda):
        return reference_relpos_attention(
            q, k, v, coords, bias_table, seq_mask, exact_lut, scale, num_tokens
        )

    coords_i32 = coords.to(torch.int32).contiguous()
    lut_i32 = exact_lut.to(torch.int32).contiguous()
    # BiasTable read in fp32 in-kernel (237*H is trivial); cast view is cheap.
    table = bias_table.to(torch.float32).contiguous()
    mask_i8 = seq_mask.to(torch.int8).contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    BLOCK_M, BLOCK_N, num_warps = _select_block(npad)
    BLOCK_D = 32  # >= HEAD_DIM=24, pad lanes zero-loaded
    grid = (triton.cdiv(s, BLOCK_M), b * h)

    _hexflash_kernel[grid](
        q, k, v, out,
        coords_i32, table, mask_i8, lut_i32,
        float(scale),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        coords_i32.stride(0), coords_i32.stride(1),
        table.stride(0), table.stride(1),
        mask_i8.stride(0), mask_i8.stride(1),
        H=h,
        S=s,
        NPAD=npad,
        NUM_TOKENS=num_tokens,
        DISK_RADIUS=BIAS_DISK_RADIUS,
        RING_MAX=BIAS_RING_MAX,
        RING_MIN=BIAS_RING_MIN,
        ON_AXIS_BASE=BIAS_ON_AXIS_BASE,
        OFF_AXIS_BASE=BIAS_OFF_AXIS_BASE,
        FAR_ROW=BIAS_FAR_ROW,
        TOK_TOK_ROW=BIAS_TOKEN_TOKEN_ROW,
        TOK_CELL_ROW=BIAS_TOKEN_CELL_ROW,
        CELL_TOK_ROW=BIAS_CELL_TOKEN_ROW,
        PAD_MASK_VALUE=PAD_KEY_MASK_VALUE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


# =============================================================================
#  FlexAttention fallback (A1-fb). Same signature; same oracle.
# =============================================================================

_FLEX_AVAILABLE: bool | None = None
_flex_attention = None
_create_block_mask = None
_BLOCKMASK_CACHE: dict[tuple, object] = {}


def _ensure_flex() -> bool:
    global _FLEX_AVAILABLE, _flex_attention, _create_block_mask
    if _FLEX_AVAILABLE is not None:
        return _FLEX_AVAILABLE
    try:
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        _flex_attention = flex_attention
        _create_block_mask = create_block_mask
        _FLEX_AVAILABLE = True
    except Exception:  # pragma: no cover - import guard
        _FLEX_AVAILABLE = False
    return _FLEX_AVAILABLE


def _pad_head_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    """(B,H,S,Dh) -> (B,H,S,target) zero-padded on the last dim."""
    dh = t.shape[-1]
    if dh == target:
        return t
    pad = t.new_zeros(*t.shape[:-1], target - dh)
    return torch.cat([t, pad], dim=-1)


def flex_attention_relpos(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,
    bias_table: torch.Tensor,
    seq_mask: torch.Tensor,
    exact_lut: torch.Tensor,
    scale: float,
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:
    """FlexAttention backend. score_mod replays the exact bias-row selection;
    mask_mod applies the pad-key mask. head_dim 24 -> 32 (zero-padded).

    NOTE: FlexAttention applies the DEFAULT scale 1/sqrt(E) internally where E
    is the (padded) head dim. We must override it to the model's 1/sqrt(24) so
    the q@kᵀ scaling matches the production path exactly — pass `scale=` to
    flex_attention rather than relying on the padded-32 default."""

    if (not _ensure_flex()) or (not q.is_cuda):
        return reference_relpos_attention(
            q, k, v, coords, bias_table, seq_mask, exact_lut, scale, num_tokens
        )

    b, h, s, dh = q.shape
    target_d = 32 if dh != 32 else dh
    qd = _pad_head_dim(q, target_d).contiguous()
    kd = _pad_head_dim(k, target_d).contiguous()
    vd = _pad_head_dim(v, target_d).contiguous()

    coords_i = coords.to(torch.int32)
    cq = coords_i[..., 0]                       # (B, Npad)
    cr = coords_i[..., 1]
    table = bias_table.to(torch.float32)
    lut = exact_lut.to(torch.int64)

    R = BIAS_DISK_RADIUS

    def score_mod(score, bb, hh, q_idx, kv_idx):
        q_tok = q_idx < num_tokens
        k_tok = kv_idx < num_tokens
        qc = q_idx - num_tokens
        kc = kv_idx - num_tokens
        qc_c = torch.clamp(qc, min=0)
        kc_c = torch.clamp(kc, min=0)
        qq = cq[bb, qc_c]
        qr = cr[bb, qc_c]
        kq = cq[bb, kc_c]
        kr = cr[bb, kc_c]
        dq = kq - qq
        dr = kr - qr
        dd = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())
        cqi = torch.clamp(dq, -R, R) + R
        cri = torch.clamp(dr, -R, R) + R
        exact_row = lut[cqi * 17 + cri]
        on_axis = (dq == 0) | (dr == 0) | ((dq + dr) == 0)
        ring_base = torch.where(
            on_axis,
            torch.full_like(dd, BIAS_ON_AXIS_BASE),
            torch.full_like(dd, BIAS_OFF_AXIS_BASE),
        )
        ring_row = ring_base + (dd - BIAS_RING_MIN)
        cell_row = torch.where(
            dd <= R,
            exact_row,
            torch.where(dd <= BIAS_RING_MAX, ring_row, torch.full_like(dd, BIAS_FAR_ROW)),
        )
        row = torch.where(
            q_tok & k_tok, torch.full_like(cell_row, BIAS_TOKEN_TOKEN_ROW),
            torch.where(
                q_tok & ~k_tok, torch.full_like(cell_row, BIAS_TOKEN_CELL_ROW),
                torch.where(
                    ~q_tok & k_tok, torch.full_like(cell_row, BIAS_CELL_TOKEN_ROW),
                    cell_row,
                ),
            ),
        )
        return score + table[row, hh].to(score.dtype)

    live = seq_mask  # (B, S) bool, True = live key (tokens always True)

    def mask_mod(bb, hh, q_idx, kv_idx):
        return live[bb, kv_idx]

    key = (b, s, seq_mask.data_ptr())
    block_mask = _BLOCKMASK_CACHE.get(key)
    if block_mask is None:
        block_mask = _create_block_mask(
            mask_mod, B=b, H=None, Q_LEN=s, KV_LEN=s, device=q.device
        )
        _BLOCKMASK_CACHE[key] = block_mask

    out = _flex_attention(
        qd, kd, vd,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=float(scale),
    )
    return out[..., :dh].contiguous()


# Backend registry — model.py routes through these names.
BACKENDS = {
    "hexflash": hexflash_attention,
    "flex": flex_attention_relpos,
    "reference": reference_relpos_attention,
}
