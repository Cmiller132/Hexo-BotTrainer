"""Fused rel-pos-bias attention Triton kernel (serve-only, opt-in).

FlashAttention-2-style online-softmax kernel specialized for the flex-pair
serve path: the per-pair bias-table ROW INDEX (the shared (B, S, S) uint8
`pair` tensor, pad-KEY columns = the appended pad row) is gathered inside the
score loop as ONE 1-byte load + one read of the tiny (BIAS_ROWS + 1, heads)
fp16 table — the same math as the flex score_mod, but in a kernel shaped for
this workload instead of flex's generic lowering:

  * head_dim is a constexpr tile dimension (main_7's d=64 hits the tensor-core
    sweet spot; flex at d=32 ran ~10 TFLOPS on Ada),
  * the key loop is bounded per batch row by `seq_lens` (tokens + last live
    cell), so fully-padded key tiles are never touched. Flex pays full S^2 for
    pad keys (measured: the WASTE_FRACTION raise regressed -43% for exactly
    this reason); here pad tail costs zero.

Skipped key tiles are numerically identical to flex's -3e4-biased pad keys:
exp(-3e4 - m) underflows to exactly 0.0 in fp32, contributing nothing to the
softmax sum. Pad keys INSIDE the seq_lens bound (non-contiguous padding,
partial tiles) still get the pad-row bias via `pair`, so correctness never
depends on padding being contiguous — only tile-skipping efficiency does.

Exposed as the `hexfield_eq::attn_pair` custom op (with a fake kernel) so the
serve torch.compile graph keeps it in-graph as an opaque call. Enabled via
HEXFIELD_TRITON_ATTN=1; model.py routes to it from RelPosAttention.forward on
the no-grad CUDA fp16 path when the block bias is a _FlexPairBias carrying
seq_lens, for head_dim in {32, 64}. Everything else falls through to flex.

Numerics: fp16 QK^T and PV tensor-core dots with fp32 accumulation, fp32
online softmax — the same accumulation class as flex_attention; output fp16.

Tile constants come from env (HEXFIELD_ATTN_BM/BN/WARPS/STAGES, read once at
import) so the bench matrix can sweep them; defaults are the measured winners.
"""

from __future__ import annotations

import math
import os
import warnings

import torch

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

# Triton's CompilationError location moves between versions; import defensively.
try:  # pragma: no cover - triton internals vary by version
    from triton.compiler.errors import CompilationError as _TritonCompileError
except Exception:  # pragma: no cover
    try:
        from triton.compiler import CompilationError as _TritonCompileError
    except Exception:
        _TritonCompileError = ()

_BM = int(os.environ.get("HEXFIELD_ATTN_BM", "64"))
_BN = int(os.environ.get("HEXFIELD_ATTN_BN", "64"))
_WARPS = int(os.environ.get("HEXFIELD_ATTN_WARPS", "4"))
_STAGES = int(os.environ.get("HEXFIELD_ATTN_STAGES", "3"))
# v2 (coords-direct) tile constants; fall back to the v1 knobs so one sweep
# variable tunes both unless explicitly split.
_BM2 = int(os.environ.get("HEXFIELD_ATTN2_BM", os.environ.get("HEXFIELD_ATTN_BM", "64")))
_BN2 = int(os.environ.get("HEXFIELD_ATTN2_BN", os.environ.get("HEXFIELD_ATTN_BN", "64")))
_WARPS2 = int(os.environ.get("HEXFIELD_ATTN2_WARPS", os.environ.get("HEXFIELD_ATTN_WARPS", "4")))
_STAGES2 = int(os.environ.get("HEXFIELD_ATTN2_STAGES", os.environ.get("HEXFIELD_ATTN_STAGES", "3")))


if HAVE_TRITON:

    # --- compile-failure fallback (cross-width eval hardening) --------------------
    # A per-arch Triton codegen edge case (see _triton_conv.py) can equally trip
    # this kernel for some head_dim under bleeding-edge triton. On ANY kernel
    # launch failure we memoize the specializing head_dim and serve it from a
    # materialized reference that reproduces the flex-pair score_mod exactly
    # (score = q·kᵀ·scale + table2[pair]; pad-key columns already carry the
    # appended pad-row's large-negative bias, so their softmax weight underflows
    # to ~0 just as the kernel's seq_lens tile-skip intends). This sits INSIDE
    # the opaque custom op so it catches the compile under both eager and the
    # serve torch.compile(dynamic=True) graph (the compile fires when the op
    # executes for a new shape, not during dynamo tracing). Keyed by head_dim D —
    # the constexpr tile dim that drives codegen; a head_dim that compiles is
    # never memoized, so its fast path is unchanged.
    _TRITON_VER = getattr(triton, "__version__", "?")
    _ATTN_FAILED: set = set()

    def _mark_attn_failed(d: int, err: Exception) -> None:
        """Record a failing head_dim and warn ONCE (called only when d is new)."""
        _ATTN_FAILED.add(d)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        warnings.warn(
            f"hexfield: triton attn_pair failed to compile for head_dim={d} "
            f"({kind}) under triton {_TRITON_VER}; using the reference path for "
            f"this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _attn_ref(q, k, v, pair, table2, seq_lens):
        """Materialized rel-pos-bias attention (flex-pair score_mod), fp16 out to
        match the custom op's fake kernel. seq_lens is unused: pad-key bias in
        `pair`/`table2` already zeros those keys in the softmax."""
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        # table2[pair] -> (B, S, S, H); move heads to (B, H, S, S).
        bias = table2.to(torch.float32)[pair.to(torch.int64)].permute(0, 3, 1, 2)
        scores = scores + bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.float())
        return out.to(torch.float16)

    @triton.jit
    def _attn_pair_kernel(
        q_ptr, k_ptr, v_ptr, pair_ptr, tab_ptr, seq_ptr, out_ptr,
        sqb, sqh, sqs,
        skb, skh, sks,
        svb, svh, svs,
        sob, soh, sos,
        spb, sps,
        H, S, scale,
        PAD_ROW: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        lo_m = pid_m * BM
        offs_m = lo_m + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < S
        o_ptrs = out_ptr + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :]

        n_keys = tl.load(seq_ptr + b)
        # Whole q tile beyond the live sequence: downstream multiplies attention
        # output by seq_mask, so pad rows only need to be finite. Store zeros.
        if lo_m >= n_keys:
            tl.store(
                o_ptrs, tl.zeros((BM, D), dtype=tl.float16), mask=m_ok[:, None]
            )
            return

        q = tl.load(
            q_ptr + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :],
            mask=m_ok[:, None],
            other=0.0,
        )
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        LOG2E: tl.constexpr = 1.4426950408889634

        for start_n in range(0, n_keys, BN):
            offs_n = start_n + tl.arange(0, BN)
            kv_ok = offs_n < n_keys
            k = tl.load(
                k_ptr + b * skb + h * skh + offs_n[:, None] * sks + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * scale
            # Bias row index; out-of-bounds -> pad row -> -3e4 -> exp == 0.
            pr = tl.load(
                pair_ptr + b * spb + offs_m[:, None] * sps + offs_n[None, :],
                mask=m_ok[:, None] & kv_ok[None, :],
                other=PAD_ROW,
            )
            qk += tl.load(tab_ptr + pr.to(tl.int32) * H + h).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_ptr + b * svb + h * svh + offs_n[:, None] * svs + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            acc += tl.dot(p.to(tl.float16), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_ok[:, None])

    @torch.library.custom_op("hexfield_eq::attn_pair", mutates_args=())
    def attn_pair(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair: torch.Tensor,
        table2: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        b, h, s, d = q.shape
        if d not in _ATTN_FAILED:
            try:
                # (B,S,H,D).transpose(1,2) views satisfy stride(-1)==1 already; the
                # kernel takes the remaining strides as-is, so no copies here.
                if q.stride(-1) != 1:
                    q = q.contiguous()
                if k.stride(-1) != 1:
                    k = k.contiguous()
                if v.stride(-1) != 1:
                    v = v.contiguous()
                pairc = pair.contiguous()
                tab = table2.to(torch.float16).contiguous()
                seq = seq_lens.to(torch.int32).contiguous()
                out = torch.empty(
                    (b, h, s, d), dtype=torch.float16, device=q.device
                )
                grid = (triton.cdiv(s, _BM), b * h)
                _attn_pair_kernel[grid](
                    q, k, v, pairc, tab, seq, out,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    pairc.stride(0), pairc.stride(1),
                    h, s, 1.0 / math.sqrt(d),
                    PAD_ROW=tab.shape[0] - 1,
                    D=d, BM=_BM, BN=_BN,
                    num_warps=_WARPS, num_stages=_STAGES,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_attn_failed(d, err)
        return _attn_ref(q, k, v, pair, table2, seq_lens)

    @attn_pair.register_fake
    def _attn_pair_fake(q, k, v, pair, table2, seq_lens):
        return q.new_empty(q.shape, dtype=torch.float16)

    # --- v2: coords-direct bias (HEXFIELD_TRITON_ATTN2) ------------------------
    # Same FA2-style online-softmax loop as attn_pair, but the per-pair bias-table
    # row is COMPUTED IN-KERNEL from the axial coords + the tiny (W*W,) u8 cell
    # LUT + the token-region row constants, instead of being gathered from a
    # precomputed (B, S, S) uint8 `pair` tensor. That tensor's once-per-forward
    # build (_build_pair_u8) measured 13.2 ms at (B=96, S=862) — more device time
    # than all three A-blocks' attention math combined — and its byte-per-pair
    # in-kernel reads are DRAM traffic (the tensor is tens-to-hundreds of MB, far
    # beyond L2). Here the geometry inputs are (B, N, 2) i32 coords, a (B, N) u8
    # live-mask and two L1-resident tables (LUT 1.2 KB, bias table ~1.4 KB), so
    # the bias costs a few integer ops + two cached gathers per tile element and
    # the S^2 memory object disappears entirely — which also unfrees the serve
    # PAIR_CEILING group-size cap (see inference.plan_groups).
    #
    # Row semantics are identical to _build_pair_u8: token/token, token/cell and
    # cell/token pairs take the three fixed rows; cell/cell pairs take
    # lut[(clamp(dq,-M,M)+M)*W + (clamp(dr,-M,M)+M)] with (dq, dr) = key - query;
    # dead-cell keys inside the seq_lens bound and partial-tile tails take the
    # appended pad row (table2[ROWS] = PAD_KEY_MASK_VALUE -> softmax weight 0).
    _ATTN2_FAILED: set = set()

    def _mark_attn2_failed(d: int, err: Exception) -> None:
        _ATTN2_FAILED.add(d)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        warnings.warn(
            f"hexfield: triton attn_coords failed to compile for head_dim={d} "
            f"({kind}) under triton {_TRITON_VER}; using the reference path for "
            f"this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _attn_coords_ref(q, k, v, coords, mask_u8, table2, lut, seq_lens):
        """Materialized reference for the coords-direct op: rebuild the pair
        row-index tensor exactly like model._build_pair_u8 (token-region rows,
        clamped-LUT cell rows, pad-KEY columns -> the appended pad row), then
        reuse _attn_ref. Slow (it materializes the S^2 objects the fast path
        exists to avoid) — correctness fallback only."""

        b, n, _ = coords.shape
        s = q.shape[2]
        nt = s - n
        w = int(math.isqrt(int(lut.numel())))
        m = (w - 1) // 2
        c = coords.to(torch.int32)
        dq = c[:, None, :, 0] - c[:, :, None, 0]  # (B, N, N) key - query
        dr = c[:, None, :, 1] - c[:, :, None, 1]
        qi = (dq.clamp(-m, m) + m).to(torch.long)
        ri = (dr.clamp(-m, m) + m).to(torch.long)
        cell = lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)
        # Token-region rows: table2 layout is [0..ROWS-1] = the expanded table,
        # row ROWS = pad. The three fixed region rows are ROWS-3/ROWS-2/ROWS-1
        # only by table convention — take them from the shared constants.
        from .constants import (
            BIAS_CELL_TOKEN_ROW,
            BIAS_TOKEN_CELL_ROW,
            BIAS_TOKEN_TOKEN_ROW,
        )

        pair = torch.full(
            (b, s, s), BIAS_TOKEN_TOKEN_ROW, dtype=torch.uint8, device=q.device
        )
        pair[:, :nt, nt:] = BIAS_TOKEN_CELL_ROW
        pair[:, nt:, :nt] = BIAS_CELL_TOKEN_ROW
        pair[:, nt:, nt:] = cell
        key_dead = torch.cat([mask_u8.new_ones(b, nt), mask_u8], dim=1) == 0
        pair = pair.masked_fill(key_dead[:, None, :], table2.shape[0] - 1)
        return _attn_ref(q, k, v, pair, table2, seq_lens)

    @triton.jit
    def _attn_coords_kernel(
        q_ptr, k_ptr, v_ptr, coord_ptr, mask_ptr, tab_ptr, lut_ptr, seq_ptr,
        out_ptr,
        sqb, sqh, sqs,
        skb, skh, sks,
        svb, svh, svs,
        sob, soh, sos,
        scb, scn,
        smb,
        H, S, scale,
        NT, M, W,
        ROW_CT, ROW_TC, ROW_TT, ROW_PAD,
        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        lo_m = pid_m * BM
        offs_m = lo_m + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < S
        o_ptrs = out_ptr + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :]

        n_keys = tl.load(seq_ptr + b)
        # Whole q tile beyond the live sequence: downstream multiplies attention
        # output by seq_mask, so pad rows only need to be finite. Store zeros.
        if lo_m >= n_keys:
            tl.store(
                o_ptrs, tl.zeros((BM, D), dtype=tl.float16), mask=m_ok[:, None]
            )
            return

        q = tl.load(
            q_ptr + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :],
            mask=m_ok[:, None],
            other=0.0,
        )
        # Query-side geometry: rows < NT are tokens (fixed bias rows); cell rows
        # index coords at (row - NT). The clamp keeps token rows' loads in
        # bounds; their (dq, dr) result is overridden by the region select.
        q_cell = tl.maximum(offs_m - NT, 0)
        q_tok = offs_m < NT
        cq = tl.load(coord_ptr + b * scb + q_cell * scn + 0, mask=m_ok, other=0)
        cr = tl.load(coord_ptr + b * scb + q_cell * scn + 1, mask=m_ok, other=0)

        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        LOG2E: tl.constexpr = 1.4426950408889634

        for start_n in range(0, n_keys, BN):
            offs_n = start_n + tl.arange(0, BN)
            kv_ok = offs_n < n_keys
            k = tl.load(
                k_ptr + b * skb + h * skh + offs_n[:, None] * sks + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * scale
            # Key-side geometry + liveness (u8 mask; tokens are always live).
            k_cell = tl.maximum(offs_n - NT, 0)
            k_tok = offs_n < NT
            kq = tl.load(coord_ptr + b * scb + k_cell * scn + 0, mask=kv_ok, other=0)
            kr = tl.load(coord_ptr + b * scb + k_cell * scn + 1, mask=kv_ok, other=0)
            k_live = tl.load(mask_ptr + b * smb + k_cell, mask=kv_ok, other=0)
            # Bias row = f(key - query): clamped-offset LUT for cell/cell,
            # fixed rows for the token regions, pad row for dead/tail keys.
            dq = kq[None, :] - cq[:, None]
            dr = kr[None, :] - cr[:, None]
            qi = tl.minimum(tl.maximum(dq, -M), M) + M
            ri = tl.minimum(tl.maximum(dr, -M), M) + M
            cell_row = tl.load(lut_ptr + qi * W + ri).to(tl.int32)
            qt = q_tok[:, None]
            kt = k_tok[None, :]
            row = tl.where(
                qt & kt,
                ROW_TT,
                tl.where(qt, ROW_TC, tl.where(kt, ROW_CT, cell_row)),
            )
            is_pad = ((k_live == 0) & (offs_n >= NT)) | (offs_n >= n_keys)
            row = tl.where(is_pad[None, :], ROW_PAD, row)
            qk += tl.load(tab_ptr + row * H + h).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_ptr + b * svb + h * svh + offs_n[:, None] * svs + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            acc += tl.dot(p.to(tl.float16), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_ok[:, None])

    @torch.library.custom_op("hexfield_eq::attn_coords", mutates_args=())
    def attn_coords(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        mask_u8: torch.Tensor,
        table2: torch.Tensor,
        lut: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        from .constants import (
            BIAS_CELL_TOKEN_ROW,
            BIAS_TOKEN_CELL_ROW,
            BIAS_TOKEN_TOKEN_ROW,
        )

        b, h, s, d = q.shape
        if d not in _ATTN2_FAILED:
            try:
                if q.stride(-1) != 1:
                    q = q.contiguous()
                if k.stride(-1) != 1:
                    k = k.contiguous()
                if v.stride(-1) != 1:
                    v = v.contiguous()
                co = coords
                if co.dtype != torch.int32:
                    co = co.to(torch.int32)
                co = co.contiguous()
                mk = mask_u8
                if mk.dtype != torch.uint8:
                    mk = mk.to(torch.uint8)
                mk = mk.contiguous()
                tab = table2.to(torch.float16).contiguous()
                lu = lut.contiguous()
                seq = seq_lens.to(torch.int32).contiguous()
                w = int(math.isqrt(int(lu.numel())))
                m = (w - 1) // 2
                nt = s - co.shape[1]
                out = torch.empty(
                    (b, h, s, d), dtype=torch.float16, device=q.device
                )
                grid = (triton.cdiv(s, _BM2), b * h)
                _attn_coords_kernel[grid](
                    q, k, v, co, mk, tab, lu, seq, out,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    co.stride(0), co.stride(1),
                    mk.stride(0),
                    h, s, 1.0 / math.sqrt(d),
                    nt, m, w,
                    BIAS_CELL_TOKEN_ROW, BIAS_TOKEN_CELL_ROW,
                    BIAS_TOKEN_TOKEN_ROW, tab.shape[0] - 1,
                    D=d, BM=_BM2, BN=_BN2,
                    num_warps=_WARPS2, num_stages=_STAGES2,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_attn2_failed(d, err)
        return _attn_coords_ref(q, k, v, coords, mask_u8, table2, lut, seq_lens)

    @attn_coords.register_fake
    def _attn_coords_fake(q, k, v, coords, mask_u8, table2, lut, seq_lens):
        return q.new_empty(q.shape, dtype=torch.float16)

else:  # pragma: no cover
    attn_pair = None
    attn_coords = None
