"""Fused gather+GEMM Triton kernel for HexNodeConv (serve-only, opt-in).

The reference HexNodeConv materializes a (B, Npad, 7*C) gathered tensor (cat a
zero row, gather, reshape) and feeds it to one GEMM. At serve shapes that
gather write+read is ~60% of the conv cost. This kernel gathers the 7 tap rows
directly into the GEMM's A-tiles (tl.dot, fp32 accumulate), so the expanded
tensor never exists; the missing-neighbour zero row becomes a masked load
(idx == Npad -> 0) and the output row mask is folded into the epilogue.

Exposed as the `hexfield::hex_conv` custom op (with a fake kernel), so the
serve torch.compile(dynamic=True) graph keeps it in-graph as an opaque call —
no graph breaks. Enabled via HEXFIELD_TRITON_CONV=1; model.py routes to it on
the no-grad CUDA path only (there is no backward), for 16-aligned channel
counts (the stem's C_in=15 keeps the reference path).

Numerics: fp16 tap rows x fp16 weights with fp32 accumulation — the same class
as the autocast cuBLAS GEMM it replaces (bias added in fp32, one final fp16
round like the reference's fp16 GEMM output). Output is fp16.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

# Triton's CompilationError location moves between versions; import defensively.
# An empty tuple makes isinstance() always False (we still fall back on ANY
# exception — this only tunes the log message).
try:  # pragma: no cover - triton internals vary by version
    from triton.compiler.errors import CompilationError as _TritonCompileError
except Exception:  # pragma: no cover
    try:
        from triton.compiler import CompilationError as _TritonCompileError
    except Exception:
        _TritonCompileError = ()

# conv+LN kernel tile knobs (bench sweeps; defaults = measured winners at
# c=192, RTX 4070 Ti, 2026-07-03: BM=64/warps=4/stages=2 runs the fused kernel
# 30-37% FASTER than plain conv + eager LN; the first-cut BM=32/warps=8 was
# the worst point in the sweep, ~neutral vs unfused).
_LN_BM = int(os.environ.get("HEXFIELD_CONVLN_BM", "64"))
_LN_WARPS = int(os.environ.get("HEXFIELD_CONVLN_WARPS", "4"))
_LN_STAGES = int(os.environ.get("HEXFIELD_CONVLN_STAGES", "2"))


if HAVE_TRITON:

    # --- compile-failure fallback (cross-width eval hardening) --------------------
    # Some channel widths trip a per-arch Triton codegen edge case (observed:
    # c=96 fails to compile _hex_conv_kernel under triton 3.7.0 / torch 2.12,
    # while c=128 compiles fine). The reference gather+GEMM path is numerically
    # equivalent, so on ANY kernel-launch failure we memoize the specializing
    # shape and serve that shape from the reference path forever after — no retry
    # of the (failing, slow) compile on every forward. This lives INSIDE the
    # custom op: under the serve torch.compile(dynamic=True) graph the op is
    # opaque and its Triton compile happens when the op EXECUTES for a new shape,
    # not during dynamo tracing, so this is the only layer that catches it under
    # both eager and compiled serve. Keyed by (C, Cout) — the dims that drive the
    # kernel's tiling/codegen; a shape that compiles (c=128) never enters a set,
    # so its fast path is byte-for-byte unchanged.
    _TRITON_VER = getattr(triton, "__version__", "?")
    _CONV_FAILED: set = set()
    _CONV_LN_FAILED: set = set()
    _CONV_FP8_FAILED: set = set()
    _CONV_LN_FP8_FAILED: set = set()

    def _mark_failed(failed: set, kernel: str, key, err: Exception) -> None:
        """Record a failing shape and warn ONCE (called only when key is new)."""
        failed.add(key)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        c, cout = key
        warnings.warn(
            f"hexfield: triton {kernel} failed to compile for C={c},Cout={cout} "
            f"({kind}) under triton {_TRITON_VER}; using the reference path for "
            f"this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _conv_ref(x, gather_idx, mask, weight, bias):
        """Reference HexNodeConv (the no-flag path in model.py), fp16 out to match
        the custom op's fake kernel."""
        b, n, c = x.shape
        cout = weight.shape[-1]
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        flat = gather_idx.to(torch.int64).reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        out = gathered @ weight.reshape(7 * c, cout) + bias
        out = out * mask.unsqueeze(-1)
        return out.to(torch.float16)

    def _conv_ln_ref(x, gather_idx, mask, weight, bias, ln_w, ln_b, eps, relu):
        """Reference conv + LayerNorm(+ReLU) + row-mask (the ConvBlock no-flag
        path). LN stats in fp32 on the conv accumulator, matching the fused
        kernel; masked rows are zeroed last (pad rows only, so valid rows match)."""
        b, n, c = x.shape
        cout = weight.shape[-1]
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        flat = gather_idx.to(torch.int64).reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        conv = gathered @ weight.reshape(7 * c, cout) + bias
        y = F.layer_norm(conv.float(), (cout,), ln_w.float(), ln_b.float(), eps)
        if relu:
            y = F.relu(y)
        y = y * mask.unsqueeze(-1)
        return y.to(torch.float16)

    @triton.jit
    def _hex_conv_kernel(
        x_ptr, idx_ptr, mask_ptr, w_ptr, bias_ptr, sc_ptr, out_ptr,
        B, Npad, C, Cout,
        IS_FP16_IN: tl.constexpr, FP8: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        n_offs = pid_n * BN + tl.arange(0, BN)  # Cout columns
        n_valid = n_offs < Cout

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for t in tl.static_range(7):
            # Row-local tap index; Npad is the missing/pad sentinel (zero row).
            idx = tl.load(idx_ptr + m_offs * 7 + t, mask=m_valid, other=Npad)
            row_ok = m_valid & (idx < Npad)
            x_row = (b_ids * Npad + idx) * C
            for k0 in tl.range(0, tl.cdiv(C, BK)):
                k_offs = k0 * BK + tl.arange(0, BK)
                k_ok = k_offs < C
                a = tl.load(
                    x_ptr + x_row[:, None] + k_offs[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                a16 = a if IS_FP16_IN else a.to(tl.float16)
                w = tl.load(
                    w_ptr + (t * C + k_offs)[:, None] * Cout + n_offs[None, :],
                    mask=k_ok[:, None] & n_valid[None, :],
                    other=0.0,
                )
                if FP8:
                    # e4m3 x e4m3 tensor-core dot (2x fp16 rate on Ada); the
                    # weight tensor is already fp8 with per-out-channel scales
                    # dequantized in the epilogue.
                    acc += tl.dot(a16.to(tl.float8e4nv), w)
                else:
                    acc += tl.dot(a16, w)

        if FP8:
            sc = tl.load(sc_ptr + n_offs, mask=n_valid, other=1.0)
            acc *= sc[None, :].to(tl.float32)
        bias = tl.load(bias_ptr + n_offs, mask=n_valid, other=0.0)
        acc += bias[None, :].to(tl.float32)
        rmask = tl.load(mask_ptr + m_offs, mask=m_valid, other=0)
        acc = tl.where(rmask[:, None] > 0, acc, 0.0)
        tl.store(
            out_ptr + m_offs[:, None] * Cout + n_offs[None, :],
            acc.to(tl.float16),
            mask=m_valid[:, None] & n_valid[None, :],
        )

    @torch.library.custom_op("hexfield::hex_conv", mutates_args=())
    def hex_conv(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
                b32 = bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
                rows = b * npad
                # Small flushes (late-game / singleton groups) need more, smaller
                # programs to keep the SMs fed; big flushes prefer the fatter tile.
                BM = 32 if rows < 32768 else 64
                BN, BK = min(128, cout), 64
                grid = (triton.cdiv(rows, BM), triton.cdiv(cout, BN))
                _hex_conv_kernel[grid](
                    x, gidx, m8, w16, b32, b32, out,  # sc_ptr unused when FP8=False
                    b, npad, c, cout,
                    IS_FP16_IN=(x.dtype == torch.float16), FP8=False,
                    BM=BM, BN=BN, BK=BK,
                    num_warps=4 if BM == 32 else 8, num_stages=3,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_FAILED, "hex_conv", key, err)
        return _conv_ref(x, gather_idx, mask, weight, bias)

    @hex_conv.register_fake
    def _hex_conv_fake(x, gather_idx, mask, weight, bias):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

    # --- conv + LayerNorm(+ReLU) + row-mask epilogue -----------------------------
    # Same fused gather+GEMM as _hex_conv_kernel, but the program owns the FULL
    # Cout row (BN >= Cout, one N-tile per program), so the ConvBlock's
    # LayerNorm -> (ReLU) -> mask epilogue runs on the fp32 accumulator before
    # the single fp16 store. Kills one full read+write of the (B, Npad, C)
    # activation per conv (the LN kernel's round-trip). LN statistics are fp32
    # over the true Cout columns; numerically the same class as the reference
    # (which LayerNorms the fp16-rounded conv output in fp32).

    @triton.jit
    def _hex_conv_ln_kernel(
        x_ptr, idx_ptr, mask_ptr, w_ptr, bias_ptr, sc_ptr, lnw_ptr, lnb_ptr,
        out_ptr,
        B, Npad, C, Cout, eps,
        IS_FP16_IN: tl.constexpr, RELU: tl.constexpr, FP8: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        n_offs = tl.arange(0, BN)  # the whole Cout row
        n_valid = n_offs < Cout

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for t in tl.static_range(7):
            idx = tl.load(idx_ptr + m_offs * 7 + t, mask=m_valid, other=Npad)
            row_ok = m_valid & (idx < Npad)
            x_row = (b_ids * Npad + idx) * C
            for k0 in tl.range(0, tl.cdiv(C, BK)):
                k_offs = k0 * BK + tl.arange(0, BK)
                k_ok = k_offs < C
                a = tl.load(
                    x_ptr + x_row[:, None] + k_offs[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                a16 = a if IS_FP16_IN else a.to(tl.float16)
                w = tl.load(
                    w_ptr + (t * C + k_offs)[:, None] * Cout + n_offs[None, :],
                    mask=k_ok[:, None] & n_valid[None, :],
                    other=0.0,
                )
                if FP8:
                    acc += tl.dot(a16.to(tl.float8e4nv), w)
                else:
                    acc += tl.dot(a16, w)

        if FP8:
            sc = tl.load(sc_ptr + n_offs, mask=n_valid, other=1.0)
            acc *= sc[None, :].to(tl.float32)
        bias = tl.load(bias_ptr + n_offs, mask=n_valid, other=0.0)
        acc += bias[None, :].to(tl.float32)
        # LayerNorm over the true Cout columns (fp32 stats on the accumulator).
        accm = tl.where(n_valid[None, :], acc, 0.0)
        mean = tl.sum(accm, 1) / Cout
        cent = tl.where(n_valid[None, :], acc - mean[:, None], 0.0)
        var = tl.sum(cent * cent, 1) / Cout
        rstd = tl.math.rsqrt(var + eps)
        lnw = tl.load(lnw_ptr + n_offs, mask=n_valid, other=0.0)
        lnb = tl.load(lnb_ptr + n_offs, mask=n_valid, other=0.0)
        y = cent * rstd[:, None] * lnw[None, :].to(tl.float32) + lnb[None, :].to(
            tl.float32
        )
        if RELU:
            y = tl.maximum(y, 0.0)
        rmask = tl.load(mask_ptr + m_offs, mask=m_valid, other=0)
        y = tl.where(rmask[:, None] > 0, y, 0.0)
        tl.store(
            out_ptr + m_offs[:, None] * Cout + n_offs[None, :],
            y.to(tl.float16),
            mask=m_valid[:, None] & n_valid[None, :],
        )

    @torch.library.custom_op("hexfield::hex_conv_ln", mutates_args=())
    def hex_conv_ln(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        relu: bool,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_LN_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
                b32 = bias.to(torch.float32).contiguous()
                lnw = ln_weight.to(torch.float32).contiguous()
                lnb = ln_bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
                rows = b * npad
                BN = triton.next_power_of_2(cout)  # whole row per program (LN needs it)
                BM, BK = _LN_BM, 64
                grid = (triton.cdiv(rows, BM),)
                _hex_conv_ln_kernel[grid](
                    x, gidx, m8, w16, b32, b32, lnw, lnb, out,  # sc_ptr unused (FP8=False)
                    b, npad, c, cout, eps,
                    IS_FP16_IN=(x.dtype == torch.float16), RELU=relu, FP8=False,
                    BM=BM, BN=BN, BK=BK,
                    num_warps=_LN_WARPS, num_stages=_LN_STAGES,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_LN_FAILED, "hex_conv_ln", key, err)
        return _conv_ln_ref(
            x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
        )

    @hex_conv_ln.register_fake
    def _hex_conv_ln_fake(
        x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
    ):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

    # --- fp8 (e4m3) variants ------------------------------------------------------
    # Ada tensor cores run e4m3 x e4m3 at 2x the fp16 rate. Weights are
    # quantized per call (trivial: 7*C*Cout elements) with per-out-channel
    # scales dequantized in the fp32 epilogue; activations are cast to fp8 in
    # registers (LayerNorm-bounded, so the e4m3 range is never clipped — only
    # mantissa precision is spent). MEDIUM numerics risk by design: gate with
    # a fresh serve-parity tolerance and the arena eval, not the 3e-3 gate.

    # Serve weights are frozen, so quantize once per weight tensor and cache.
    # The entry keeps a strong ref to the weight (id() can't alias a collected
    # tensor) and re-quantizes if the weight is ever mutated in place.
    _W8_CACHE: dict = {}

    def _w8_scales(weight: torch.Tensor, c: int, cout: int):
        key = id(weight)
        ent = _W8_CACHE.get(key)
        if ent is not None and ent[0] is weight and ent[1] == weight._version:
            return ent[2], ent[3]
        w = weight.reshape(7 * c, cout).to(torch.float32)
        sc = (w.abs().amax(dim=0) / 448.0).clamp(min=1e-12)
        w8 = (w / sc).to(torch.float8_e4m3fn).contiguous()
        sc = sc.contiguous()
        _W8_CACHE[key] = (weight, weight._version, w8, sc)
        return w8, sc

    @torch.library.custom_op("hexfield::hex_conv_fp8", mutates_args=())
    def hex_conv_fp8(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_FP8_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w8, sc = _w8_scales(weight, c, cout)
                b32 = bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
                rows = b * npad
                BM = 32 if rows < 32768 else 64
                BN, BK = min(128, cout), 64
                grid = (triton.cdiv(rows, BM), triton.cdiv(cout, BN))
                _hex_conv_kernel[grid](
                    x, gidx, m8, w8, b32, sc, out,
                    b, npad, c, cout,
                    IS_FP16_IN=(x.dtype == torch.float16), FP8=True,
                    BM=BM, BN=BN, BK=BK,
                    num_warps=4 if BM == 32 else 8, num_stages=3,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_FP8_FAILED, "hex_conv_fp8", key, err)
        # fp16 reference: dropping fp8 on failure only improves numerics.
        return _conv_ref(x, gather_idx, mask, weight, bias)

    @hex_conv_fp8.register_fake
    def _hex_conv_fp8_fake(x, gather_idx, mask, weight, bias):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

    @torch.library.custom_op("hexfield::hex_conv_ln_fp8", mutates_args=())
    def hex_conv_ln_fp8(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        relu: bool,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_LN_FP8_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w8, sc = _w8_scales(weight, c, cout)
                b32 = bias.to(torch.float32).contiguous()
                lnw = ln_weight.to(torch.float32).contiguous()
                lnb = ln_bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
                rows = b * npad
                BN = triton.next_power_of_2(cout)
                BM, BK = _LN_BM, 64
                grid = (triton.cdiv(rows, BM),)
                _hex_conv_ln_kernel[grid](
                    x, gidx, m8, w8, b32, sc, lnw, lnb, out,
                    b, npad, c, cout, eps,
                    IS_FP16_IN=(x.dtype == torch.float16), RELU=relu, FP8=True,
                    BM=BM, BN=BN, BK=BK,
                    num_warps=_LN_WARPS, num_stages=_LN_STAGES,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_LN_FP8_FAILED, "hex_conv_ln_fp8", key, err)
        # fp16 reference: dropping fp8 on failure only improves numerics.
        return _conv_ln_ref(
            x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
        )

    @hex_conv_ln_fp8.register_fake
    def _hex_conv_ln_fp8_fake(
        x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
    ):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

else:  # pragma: no cover
    hex_conv = None
    hex_conv_ln = None
    hex_conv_fp8 = None
    hex_conv_ln_fp8 = None
