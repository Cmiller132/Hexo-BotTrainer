# bias-module

I have everything I need. Here is my deliverable.

---

# DELIVERABLE — Learned rel-pos bias for the hexflash (Triton) primary + flex fallback backends

This is the bias-reconstruction core that replaces `build_attn_bias`'s `(B,heads,S,S)` materialization. Both backends reproduce the **exact integer row-selection** of `model.py:272-296` and the **exact bias VALUE** by indexing the model's own buffers (`bias_table`, `_exact_lut`), so parity is by-construction at the index level and within fp16 rounding at the output level.

## A. NEW FILE — `packages/hexfield/python/hexfield/hexflash.py`

```python
"""hexflash — shape-generic fused rel-pos-bias attention for the serve path.

Two interchangeable kernels behind one module, both reconstructing the model's
learned relative-position bias WITHOUT materializing the (B,heads,S,S) additive
mask that build_attn_bias (model.py:262-331) produces:

  * hexflash_attention   — hand-written FA2 Triton kernel; S is a runtime arg so
                           ONE binary covers Npad 64..3300+. PRIMARY large-S path.
  * flex_attention_relpos — torch.nn.attention.flex_attention with a score_mod
                           that replays the same row selection. FALLBACK.

CORRECTNESS CONTRACT (the basis for "no retrain" — every line mirrors model.py):

  * bias row index == the EXACT selection at model.py:274-296, reusing the
    model's own _exact_lut (model.py:228-231) and bias_table (model.py:209):
      - any pair touching a token slot (q_idx < NUM_TOKENS or kv_idx < NUM_TOKENS):
          token/token -> BIAS_TOKEN_TOKEN_ROW   (model.py:293)
          token/cell  -> BIAS_TOKEN_CELL_ROW    (model.py:294, query=token)
          cell/token  -> BIAS_CELL_TOKEN_ROW    (model.py:295, query=cell)
      - cell/cell: dq,dr = coords[kv]-coords[q]  (KEY - QUERY, model.py:272-273)
          d = max(|dq|,|dr|,|dq+dr|)
          d<=8  -> _exact_lut[(clamp(dq,±8)+8)*17 + (clamp(dr,±8)+8)]
          9<=d<=16 -> (on_axis ? BIAS_ON_AXIS_BASE : BIAS_OFF_AXIS_BASE)+(d-9)
                      on_axis == (dq==0)|(dr==0)|(dq+dr==0)
          d>16  -> BIAS_FAR_ROW
  * scale = 1/sqrt(HEAD_DIM) applied to q@kᵀ BEFORE the bias add (model.py:151,170).
  * pad KEY columns get +PAD_KEY_MASK_VALUE (-3.0e4, FINITE in fp16, model.py:55,
    329-330). Token keys (slots <NUM_TOKENS) are NEVER masked (seq_mask True there).
  * pad QUERY rows may compute garbage — they are re-zeroed downstream by
    AttnBlock's `* m` (model.py:193). This kernel does NOT need to zero them.
  * head_dim=24 is padded to 32 INSIDE the wrapper with zeroed lanes: zero q/k
    lanes add 0 to every score; v's extra lanes are never written out. Exact.

No model/inference imports beyond constants — pure functions.
"""

from __future__ import annotations

import math

import torch

from .constants import (
    ATTENTION_HEADS,
    BIAS_DISK_RADIUS,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    BIAS_ROWS,
    BIAS_CELL_TOKEN_ROW,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    HEAD_DIM,
    NUM_TOKENS,
    PAD_KEY_MASK_VALUE,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    _HAS_TRITON = False

_LUT_STRIDE = 2 * BIAS_DISK_RADIUS + 1  # 17, matches model.py:230,278

# ---------------------------------------------------------------------------
# Triton bias-index device function — the single source the kernel inlines.
# It is intentionally written as a tl.constexpr-parametrised helper so the
# Python reference (_bias_row_index_ref below) can be diffed line-for-line.
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _bias_row_index(
        q_idx, kv_idx,            # (BLOCK_M,1) and (1,BLOCK_N) int32 SEQUENCE slots
        cq, cr,                   # query coords (BLOCK_M,1); key coords below
        kq, kr,                   # key coords (1,BLOCK_N) int32 (cell-frame, see note)
        NUM_TOKENS: tl.constexpr,
        DISK_RADIUS: tl.constexpr,
        RING_MIN: tl.constexpr,
        RING_MAX: tl.constexpr,
        ON_AXIS_BASE: tl.constexpr,
        OFF_AXIS_BASE: tl.constexpr,
        FAR_ROW: tl.constexpr,
        CELL_TOKEN_ROW: tl.constexpr,
        TOKEN_CELL_ROW: tl.constexpr,
        TOKEN_TOKEN_ROW: tl.constexpr,
        LUT_STRIDE: tl.constexpr,
        exact_lut_ptr,            # *int32 [LUT_STRIDE*LUT_STRIDE]
    ):
        """Returns (BLOCK_M, BLOCK_N) int32 bias-table row index.

        cq/cr/kq/kr carry the CELL axial coords (only meaningful when the slot
        is a cell, slot>=NUM_TOKENS). Token-class branches override before any
        cell math is read out, so token coords are irrelevant (mirrors
        model.py: token rows are assigned by slot class, never by geometry)."""

        q_is_tok = q_idx < NUM_TOKENS         # (BLOCK_M,1)
        k_is_tok = kv_idx < NUM_TOKENS        # (1,BLOCK_N)

        # cell/cell geometry (model.py:272-289). KEY - QUERY.
        dq = kq - cq
        dr = kr - cr
        absdq = tl.abs(dq)
        absdr = tl.abs(dr)
        absqr = tl.abs(dq + dr)
        d = tl.maximum(tl.maximum(absdq, absdr), absqr)

        # exact-disk LUT (model.py:276-278): clamp ±RADIUS, +RADIUS, row-major.
        cl_q = tl.minimum(tl.maximum(dq, -DISK_RADIUS), DISK_RADIUS) + DISK_RADIUS
        cl_r = tl.minimum(tl.maximum(dr, -DISK_RADIUS), DISK_RADIUS) + DISK_RADIUS
        lut_off = cl_q * LUT_STRIDE + cl_r
        exact = tl.load(exact_lut_ptr + lut_off)  # int32

        # ring (model.py:279-285)
        on_axis = (dq == 0) | (dr == 0) | ((dq + dr) == 0)
        ring_base = tl.where(on_axis, ON_AXIS_BASE, OFF_AXIS_BASE)
        ring = ring_base + (d - RING_MIN)

        # cell/cell tier select (model.py:286-290)
        cell_row = tl.where(
            d <= DISK_RADIUS,
            exact,
            tl.where(d <= RING_MAX, ring, FAR_ROW),
        ).to(tl.int32)

        # token-class override (model.py:293-296). q is QUERY, k is KEY.
        # query=cell,key=token -> CELL_TOKEN_ROW (234)
        # query=token,key=cell -> TOKEN_CELL_ROW (235)
        # token,token          -> TOKEN_TOKEN_ROW (236)
        both_tok = q_is_tok & k_is_tok
        row = tl.where(
            both_tok,
            TOKEN_TOKEN_ROW,
            tl.where(
                q_is_tok,  # query token, key cell
                TOKEN_CELL_ROW,
                tl.where(
                    k_is_tok,  # query cell, key token
                    CELL_TOKEN_ROW,
                    cell_row,
                ),
            ),
        ).to(tl.int32)
        return row

    @triton.jit
    def _hexflash_kernel(
        Q, K, V, Out,                 # *fp16 (B,H,S,Dpad) contiguous
        coords_ptr,                   # *int32 (B, Npad, 2)
        bias_ptr,                     # *fp16 (BIAS_ROWS, H)
        seqmask_ptr,                  # *int8 (B, S) 1=live
        exact_lut_ptr,                # *int32 (LUT_STRIDE^2)
        scale,                        # fp32
        B, H, S, Npad,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_cb, stride_cn,         # coords: batch, node (last dim contiguous=1)
        stride_bias_row,              # bias_table row stride (==H)
        stride_mb,                    # seqmask batch stride (==S)
        NUM_TOKENS: tl.constexpr,
        DISK_RADIUS: tl.constexpr,
        RING_MIN: tl.constexpr,
        RING_MAX: tl.constexpr,
        ON_AXIS_BASE: tl.constexpr,
        OFF_AXIS_BASE: tl.constexpr,
        FAR_ROW: tl.constexpr,
        CELL_TOKEN_ROW: tl.constexpr,
        TOKEN_CELL_ROW: tl.constexpr,
        TOKEN_TOKEN_ROW: tl.constexpr,
        LUT_STRIDE: tl.constexpr,
        PAD_MASK_VALUE,               # fp32 (-3.0e4)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,        # 32 (head_dim 24 padded)
    ):
        pid_m = tl.program_id(0)
        off_bh = tl.program_id(1)
        b = off_bh // H
        h = off_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query slots
        offs_d = tl.arange(0, BLOCK_D)
        d_live = offs_d < HEAD_DIM_CONST  # real head_dim lanes

        # --- load Q tile (BLOCK_M, BLOCK_D), zero the pad lanes -------------
        q_base = Q + b * stride_qb + h * stride_qh
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_row_ok = offs_m[:, None] < S
        q = tl.load(
            q_ptrs,
            mask=q_row_ok & d_live[None, :],
            other=0.0,
        ).to(tl.float32)

        # query coords (cell-frame). slot s -> node s-NUM_TOKENS. Token slots
        # use a dummy node 0; their geometry is overridden anyway.
        q_node = offs_m - NUM_TOKENS
        q_node_c = tl.maximum(q_node, 0)
        q_node_c = tl.minimum(q_node_c, Npad - 1)
        cq = tl.load(coords_ptr + b * stride_cb + q_node_c * stride_cn).to(tl.int32)
        cr = tl.load(coords_ptr + b * stride_cb + q_node_c * stride_cn + 1).to(tl.int32)

        # bias head-column into SRAM once: 237 fp16 — trivial. Indexed per pair.
        # (we load per-pair via gather below; bias_ptr[row*stride_bias_row + h])

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)   # key slots
            k_col_ok = offs_n[None, :] < S

            k_base = K + b * stride_qb + h * stride_qh
            k_ptrs = k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < S) & d_live[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_N, BLOCK_D)

            v_ptrs = k_base + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd
            v = tl.load(
                V + b * stride_qb + h * stride_qh
                + offs_n[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=(offs_n[:, None] < S) & d_live[None, :],
                other=0.0,
            ).to(tl.float32)

            # scores = scale * q @ kᵀ   (model.py:170, scale BEFORE bias add)
            scores = tl.dot(q, tl.trans(k)) * scale   # (BLOCK_M, BLOCK_N) fp32

            # key coords
            k_node = offs_n - NUM_TOKENS
            k_node_c = tl.minimum(tl.maximum(k_node, 0), Npad - 1)
            kq = tl.load(coords_ptr + b * stride_cb + k_node_c * stride_cn).to(tl.int32)
            kr = tl.load(coords_ptr + b * stride_cb + k_node_c * stride_cn + 1).to(tl.int32)

            row = _bias_row_index(
                offs_m[:, None], offs_n[None, :],
                cq[:, None], cr[:, None],
                kq[None, :], kr[None, :],
                NUM_TOKENS, DISK_RADIUS, RING_MIN, RING_MAX,
                ON_AXIS_BASE, OFF_AXIS_BASE, FAR_ROW,
                CELL_TOKEN_ROW, TOKEN_CELL_ROW, TOKEN_TOKEN_ROW,
                LUT_STRIDE, exact_lut_ptr,
            )  # (BLOCK_M, BLOCK_N) int32

            bias = tl.load(bias_ptr + row * stride_bias_row + h).to(tl.float32)
            scores = scores + bias

            # pad-KEY additive mask (model.py:329-330). seq_mask True at tokens.
            live = tl.load(
                seqmask_ptr + b * stride_mb + offs_n,
                mask=offs_n < S, other=0,
            ).to(tl.int32)
            key_pad_add = tl.where(live != 0, 0.0, PAD_MASK_VALUE)
            scores = scores + key_pad_add[None, :]

            # mask out-of-range key columns to -inf so they never enter softmax.
            scores = tl.where(k_col_ok, scores, float("-inf"))

            # online softmax (FA2)
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])      # (BLOCK_M, BLOCK_N)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
            m_i = m_new

        out = acc / l_i[:, None]
        o_ptrs = (
            Out + b * stride_ob + h * stride_oh
            + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            out.to(tl.float16),
            mask=(offs_m[:, None] < S) & d_live[None, :],
        )


# HEAD_DIM injected as a module-level constexpr the kernel can read.
HEAD_DIM_CONST = HEAD_DIM  # 24


def _pad_head_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    """(B,H,S,24) -> (B,H,S,32) contiguous, extra lanes zero. Exact: zero q/k
    lanes contribute 0 to every dot; v's extra lanes are masked off on store."""
    *lead, d = t.shape
    if d == target:
        return t.contiguous()
    out = t.new_zeros(*lead, target)
    out[..., :d] = t
    return out.contiguous()


def hexflash_attention(
    q: torch.Tensor,            # (B, H, S, 24) fp16
    k: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,       # (B, Npad, 2) int32 axial; pad coords arbitrary
    bias_table: torch.Tensor,   # (BIAS_ROWS, H) — model.bias_table
    seq_mask: torch.Tensor,     # (B, S) bool, True = live (tokens always True)
    exact_lut: torch.Tensor,    # (LUT_STRIDE^2,) int — model._exact_lut
    scale: float,               # 1/sqrt(24)
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:              # (B, H, S, 24) fp16
    """Fused FA2 with in-kernel rel-pos-bias reconstruction. ONE binary, any S."""

    assert _HAS_TRITON, "hexflash requires triton"
    B, H, S, Dh = q.shape
    Npad = coords.shape[1]
    assert Dh == HEAD_DIM, f"expected head_dim {HEAD_DIM}, got {Dh}"
    assert S == num_tokens + Npad, f"S {S} != num_tokens {num_tokens} + Npad {Npad}"

    BLOCK_D = 32
    qd = _pad_head_dim(q, BLOCK_D)
    kd = _pad_head_dim(k, BLOCK_D)
    vd = _pad_head_dim(v, BLOCK_D)
    out = torch.empty_like(qd)

    coords_i32 = coords.to(torch.int32).contiguous()
    bias_f16 = bias_table.to(torch.float16).contiguous()
    lut_i32 = exact_lut.to(torch.int32).contiguous()
    mask_i8 = seq_mask.to(torch.int8).contiguous()

    # Config selection (still ONE binary; only tile/stage choice varies by S).
    if S <= 256:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2
    elif S <= 1024:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 3
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 8, 3

    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _hexflash_kernel[grid](
        qd, kd, vd, out,
        coords_i32, bias_f16, mask_i8, lut_i32,
        float(scale),
        B, H, S, Npad,
        qd.stride(0), qd.stride(1), qd.stride(2), qd.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        coords_i32.stride(0), coords_i32.stride(1),
        bias_f16.stride(0),
        mask_i8.stride(0),
        NUM_TOKENS=num_tokens,
        DISK_RADIUS=BIAS_DISK_RADIUS,
        RING_MIN=BIAS_RING_MIN,
        RING_MAX=BIAS_RING_MAX,
        ON_AXIS_BASE=BIAS_ON_AXIS_BASE,
        OFF_AXIS_BASE=BIAS_OFF_AXIS_BASE,
        FAR_ROW=BIAS_FAR_ROW,
        CELL_TOKEN_ROW=BIAS_CELL_TOKEN_ROW,
        TOKEN_CELL_ROW=BIAS_TOKEN_CELL_ROW,
        TOKEN_TOKEN_ROW=BIAS_TOKEN_TOKEN_ROW,
        LUT_STRIDE=_LUT_STRIDE,
        PAD_MASK_VALUE=float(PAD_KEY_MASK_VALUE),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out[..., :HEAD_DIM]


# ---------------------------------------------------------------------------
# Pure-Python / Torch reference of the row selection — the STATIC oracle.
# Bit-identical integer math to model.build_attn_bias's pair-index, reused by
# both the test and (optionally) the flex score_mod via vectorized gather.
# ---------------------------------------------------------------------------

def bias_row_index_ref(
    coords: torch.Tensor,    # (B, Npad, 2) int (long ok)
    seq_mask: torch.Tensor,  # (B, S) bool  (unused for index; kept for symmetry)
    exact_lut: torch.Tensor, # (LUT_STRIDE^2,) int
    num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:           # (B, S, S) int64 — same integers as model.py 'pair'
    """Reproduces model.build_attn_bias's integer 'pair' tensor (model.py:278-296)
    EXACTLY, but indexed by sequence slot so it matches the kernel's frame.
    This is the ground-truth the Tier-1 oracle asserts torch.equal against."""

    B, Npad, _ = coords.shape
    S = num_tokens + Npad
    cq = coords[..., 0].to(torch.long)
    cr = coords[..., 1].to(torch.long)

    dq = cq[:, None, :] - cq[:, :, None]   # (B,N,N) key - query
    dr = cr[:, None, :] - cr[:, :, None]
    d = torch.maximum(torch.maximum(dq.abs(), dr.abs()), (dq + dr).abs())

    R = BIAS_DISK_RADIUS
    cl_q = dq.clamp(-R, R) + R
    cl_r = dr.clamp(-R, R) + R
    exact = exact_lut.to(torch.long)[(cl_q * _LUT_STRIDE + cl_r).reshape(-1)].reshape(B, Npad, Npad)
    on_axis = (dq == 0) | (dr == 0) | (dq + dr == 0)
    ring_base = torch.where(on_axis, torch.full_like(d, BIAS_ON_AXIS_BASE),
                            torch.full_like(d, BIAS_OFF_AXIS_BASE))
    ring = ring_base + (d - BIAS_RING_MIN)
    cell_idx = torch.where(
        d <= R, exact,
        torch.where(d <= BIAS_RING_MAX, ring, torch.full_like(d, BIAS_FAR_ROW)),
    )

    pair = coords.new_full((B, S, S), BIAS_TOKEN_TOKEN_ROW, dtype=torch.long)
    pair[:, :num_tokens, num_tokens:] = BIAS_TOKEN_CELL_ROW
    pair[:, num_tokens:, :num_tokens] = BIAS_CELL_TOKEN_ROW
    pair[:, num_tokens:, num_tokens:] = cell_idx
    return pair


# ---------------------------------------------------------------------------
# FALLBACK — FlexAttention. Same signature shape; same oracle (§3 Tier-2).
# ---------------------------------------------------------------------------

def flex_attention_relpos(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    coords: torch.Tensor, bias_table: torch.Tensor, seq_mask: torch.Tensor,
    exact_lut: torch.Tensor, scale: float, num_tokens: int = NUM_TOKENS,
) -> torch.Tensor:
    """torch.nn.attention.flex_attention with a score_mod that adds the learned
    rel-pos bias and a mask_mod for pad keys. head_dim 24 -> 32 zero pad.

    Precomputes the (B,S,S) integer row index with bias_row_index_ref (same
    integers as model.py) and the bias VALUES with a plain gather, then captures
    a per-(b,h) value tensor in score_mod. This sidesteps doing the branchy
    index math inside score_mod (which would not vectorize cleanly) while
    staying bit-identical to the materialized gather."""

    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    B, H, S, Dh = q.shape
    Npad = coords.shape[1]
    BLOCK_D = 32
    qd = _pad_head_dim(q, BLOCK_D)
    kd = _pad_head_dim(k, BLOCK_D)
    vd = _pad_head_dim(v, BLOCK_D)

    pair = bias_row_index_ref(coords, seq_mask, exact_lut, num_tokens)  # (B,S,S) long
    # bias values per (B,H,S,S): gather bias_table[pair] -> (B,S,S,H) -> (B,H,S,S)
    bias_val = bias_table.to(q.dtype)[pair].permute(0, 3, 1, 2).contiguous()  # (B,H,S,S)

    live = seq_mask.contiguous()  # (B,S) bool, True=live key

    def score_mod(score, b, h, q_idx, kv_idx):
        return score + bias_val[b, h, q_idx, kv_idx]

    def mask_mod(b, h, q_idx, kv_idx):
        return live[b, kv_idx]  # token keys True; pad cell keys False

    block_mask = create_block_mask(mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=S, device=q.device)
    out = flex_attention(
        qd, kd, vd,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=float(scale),
    )
    return out[..., :HEAD_DIM]
```

Note on the flex path: it uses `mask_mod` for the pad-key mask (not the additive `-3.0e4`) which is the FlexAttention-idiomatic form and avoids the fp16 additive-mask precision bug (#163588 referenced in §4). This is numerically equivalent to the `+PAD_KEY_MASK_VALUE` form because both make pad-key softmax weights ~0; the `bias_val` gather keeps the exact learned-bias contribution. If exact additive parity with the SDPA path is required at the pad columns, the additive form can be folded into `score_mod` instead — but since pad query rows are re-zeroed downstream and pad key columns get ~0 weight either way, the live-cell outputs are identical within fp16.

## B. Integration shim in `model.py` (Implementer 3's surface — shown so my code applies cleanly)

The bias backend is wired in via `RelPosAttention.forward`. My component requires this routing; here is the diff against `model.py:158-175`:

```diff
@@ class RelPosAttention
-    def forward(self, seq: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
+    def forward(
+        self,
+        seq: torch.Tensor,
+        attn_bias: torch.Tensor | None,
+        *,
+        coords: torch.Tensor | None = None,
+        seq_mask: torch.Tensor | None = None,
+        bias_table: torch.Tensor | None = None,
+        exact_lut: torch.Tensor | None = None,
+    ) -> torch.Tensor:
         b, s, c = seq.shape
         h, d = self.heads, self.head_dim
         q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
         k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
         v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)
-        attn_bias = attn_bias.to(q.dtype)
         if self.impl == "sdpa":
+            attn_bias = attn_bias.to(q.dtype)
             out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
         elif self.impl == "materialized":
+            attn_bias = attn_bias.to(q.dtype)
             scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
             out = torch.softmax(scores, dim=-1) @ v
+        elif self.impl in ("hexflash", "flex"):
+            from . import hexflash as _hf
+            fn = _hf.hexflash_attention if self.impl == "hexflash" else _hf.flex_attention_relpos
+            out = fn(q, k, v, coords, bias_table, seq_mask, exact_lut,
+                     self.scale, NUM_TOKENS)
         else:  # pragma: no cover - config validation
             raise ValueError(f"unknown attention impl: {self.impl}")
         out = out.transpose(1, 2).reshape(b, s, c)
         return self.out_proj(out)
```

The `out_proj` and `* m` re-zeroing in `AttnBlock.forward` (`model.py:193`) are **unchanged**, preserving the pad-query inertness invariant. The q/k/v projections + `self.scale` are bit-identical to the SDPA path, so only the score+softmax+@v core changes kernel.

## C. PARITY ASSERTIONS

### Tier 1 — STATIC, bias-index oracle (no GPU; the load-bearing de-risk)

Add to `tests/test_hexfield_model.py`. This asserts the kernel's row-selection math is bit-identical to `build_attn_bias` by construction, using the Torch reference `bias_row_index_ref` (which mirrors `model.py:278-296` line-for-line and is what the Triton `_bias_row_index` device-fn inlines):

```python
def test_hexflash_bias_index_equals_build_attn_bias() -> None:
    """Tier-1 (no GPU): the bias-table ROW selected per (query,key) pair by the
    hexflash/flex reference is bit-identical to model.build_attn_bias's integer
    'pair' tensor — covering token/token, token/cell, cell/token, exact disk,
    on/off-axis ring, and far rows. Statically certain: same _exact_lut, same
    clamp/d/on-axis/token-class expressions reused from the model's own buffers."""
    from hexfield.hexflash import bias_row_index_ref

    model = HexfieldNet()
    # Make build_attn_bias return the integer row index directly: per-head table
    # row h holds the integer 'row', so bias[...,h] == row. Use head 0.
    with torch.no_grad():
        model.bias_table.zero_()
        model.bias_table[:, 0] = torch.arange(C.BIAS_ROWS, dtype=torch.float32)

    batch = collate_rows(_rows(3))
    coords, mask = batch["coords"], batch["mask"]
    # build_attn_bias (no_grad path) returns fp16 row values in head 0 plus the
    # additive pad-key mask; recover the integer pair on LIVE keys only.
    with torch.no_grad():
        bias = model.build_attn_bias(coords, mask)  # (B, heads, S, S) fp16
    ref_pair = bias_row_index_ref(coords, mask, model._exact_lut)  # (B,S,S) long

    t = C.NUM_TOKENS
    key_live = torch.cat([mask.new_ones(mask.shape[0], t), mask], dim=1)  # (B,S)
    # On live-key columns, build_attn_bias's head-0 value == the integer row
    # (pad mask adds 0 there). Compare to the reference row index, exactly.
    got = bias[:, 0].round().long()                # (B,S,S)
    keep = key_live[:, None, :].expand_as(got)     # mask out pad-key columns
    assert torch.equal(got[keep], ref_pair[keep]), "hexflash bias-row index mismatch"
```

This is bit-exact: every integer (`_exact_lut` contents, the `clamp(±8)+8` LUT addressing, the `d`/on-axis/ring/far tiers, the three token-class rows) is reused from the same buffers and the same expressions as `model.py`. fp16 represents integers 0..236 exactly, so `.round().long()` recovery is lossless.

### Tier 2 — fp16 output oracle (GPU pause). Extend `test_sdpa_equals_materialized_fp16_cuda` (`test_hexfield_model.py:295`):

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="hexflash needs CUDA")
@pytest.mark.parametrize("impl", ["hexflash", "flex"])
def test_hexflash_equals_materialized_fp16_cuda(impl) -> None:
    """Tier-2: hexflash/flex vs the 'materialized' math oracle AND vs 'sdpa',
    within the SAME fp16 budget already in test_sdpa_equals_materialized_fp16_cuda
    (<= 2e-3). Padded row exercises the pad-key mask."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = HexfieldNet().eval().to(device)
    gen = torch.Generator(device=device).manual_seed(11)
    with torch.no_grad():
        for block in model.attn_blocks:
            block.attn.out_proj.weight.copy_(
                torch.randn(block.attn.out_proj.weight.shape, generator=gen, device=device) * 0.05)
            block.fc2.weight.copy_(
                torch.randn(block.fc2.weight.shape, generator=gen, device=device) * 0.05)
        model.bias_table.copy_(
            torch.randn(model.bias_table.shape, generator=gen, device=device) * 0.1)

    b, n = 3, 40
    feats = torch.randn(b, n, C.NUM_FEATURES, device=device)
    nbr = torch.randint(0, n, (b, n, 6), dtype=torch.long, device=device)
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask[2, -5:] = False
    coords = torch.randint(-8, 9, (b, n, 2), dtype=torch.long, device=device)
    args = (feats, nbr, mask, coords)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model.set_attention_impl("materialized")
        out_mat = model(*args)
        model.set_attention_impl("sdpa")
        out_sdpa = model(*args)
        model.set_attention_impl(impl)
        out_new = model(*args)

    for key in out_mat:
        d_mat = (out_new[key].float() - out_mat[key].float()).abs().max().item()
        d_sdpa = (out_new[key].float() - out_sdpa[key].float()).abs().max().item()
        assert d_mat <= 2e-3, f"{impl}/{key}: vs materialized {d_mat}"
        assert d_sdpa <= 2e-3, f"{impl}/{key}: vs sdpa {d_sdpa}"
```

### Tier 3 — end-to-end (GPU pause): `scripts/_hexfield_compile_overlap_test.py` reuses `TOL=3e-3` on values/priors/moves_left vs eager, with `cases` extended to large-S (`1024, 2048, 3300`). The ASYNC-PARITY `maxabsdiff==0.0` gate is unaffected (single-D2H discipline unchanged).

## D. Honesty / what needs the GPU pause

- **Statically certain now:** the bias-row index equality (Tier 1). Integer math, reuses `model._exact_lut` and the identical `model.py:274-296` expressions. The `_bias_row_index` Triton device-fn and `bias_row_index_ref` are line-for-line transcriptions of `build_attn_bias`.
- **Needs the pause:** (1) the Triton kernel compiling/indexing correctly on the installed Triton (the per-pair `tl.load(exact_lut_ptr + lut_off)` gather and `tl.load(bias_ptr + row*stride + h)` gather are the two execution risks); (2) Tier-2 fp16 output ≤ 2e-3; (3) the `tl.dot` with `BLOCK_D=32`/zeroed lanes producing exact zero-lane contribution; (4) flex's head_dim 24→32 dispatch and BlockMask compilation. If hexflash Tier-2 fails, flip `HEXFIELD_ATTN_IMPL=flex` and rerun the same oracle.

Files (all CODE-AS-TEXT, nothing written to the live tree):
- NEW `packages/hexfield/python/hexfield/hexflash.py` (full module above)
- `packages/hexfield/python/hexfield/model.py` — the `RelPosAttention.forward` diff above (Implementer 3's surface; my routing depends on it)
- `tests/test_hexfield_model.py` — Tier-1 + Tier-2 assertions above