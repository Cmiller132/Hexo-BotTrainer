"""Fused Triton ray-tap aggregation for the TRAIN stream (fwd + bwd).

The eager K2 path (`_raytap._RayTapTaps`) is memory-bounded but materializes,
per equipped conv per direction, several (B, N, RAY_REACH, C) fp32 tensors in
forward AND backward (masked gather, effective-alpha broadcast, product), plus
scatter_adds for the table grads. At live main_3 shapes (B=59, Npad=512,
C=192, 10 equipped convs) that island is ~85% of the training step's device
time and ~5k kernel launches per microbucket (profiled 2026-07-13), all of it
DRAM-bandwidth-bound elementwise/gather traffic.

This module fuses the whole aggregation into two Triton kernels behind one
custom op with registered autograd, so the compiled trainer keeps it in-graph
(no per-conv graph break) and NO (B, N, RAY_REACH, C) intermediate ever
exists:

  rt_taps7_train(x, idx_taps, reach, alpha_full, O_full, P_full, corb)
      -> (B, N, 7, C) tapped GEMM input, [center ; 6 direction taps]

  * forward: one kernel; per (row, tap) it walks k = 1..5, loads the source
    row only when visible to either side, and accumulates
    ``x_src * vis * (alpha_k + O[reach_own, k] + P[reach_opp, k])`` in fp32.
    The center tap is copied in-kernel (the model's `cat` disappears).
  * backward: ONE kernel, scatter-formulated exactly like the eager K2
    reference (each program owns query rows, re-gathers the masked x rows,
    and atomically scatters coeff*vis*go into the tap source rows of grad_x
    while bucketing contrib = xg*vis*go into the (6, 5, C) table-grad
    buffers). A gather-formulated grad_x (opposite-tap bijection) was tried
    first and ran 40-100x slower — its per-(t, k) serial dependency chain
    (idx gather -> reach gather -> table select) defeats latency hiding.
    ``grad_alpha`` is derived as ``grad_O.sum(0)`` in LUT mode (every row
    has exactly one own-reach state, so the O-buckets partition the alpha
    sum); the no-LUT specialization accumulates grad_alpha directly.

Numerics: identical value class to `_raytap.ray_tap_taps_naive` — the same
masked-gather × effective-alpha × visibility products in the same k order,
fp32 accumulation, and the visibility factor is exactly {0.0, 1.0} so the
association order is exact. Table-grad accumulation order differs from the
eager scatter_add (both are atomic and unordered). The T8-class oracle in
tests/test_hexfield_eq_raytap_triton_train.py pins fwd + all grads against
the naive reference.

Fallback: any Triton compile/launch failure memoizes the (C, has_lut) key and
routes that shape to the eager reference math forever after (the serve
kernels' hardening pattern). HEXFIELD_TRITON_RAYTAP_TRAIN=0 disables the
whole path (model.py then uses the eager K2 island unchanged).
"""

from __future__ import annotations

import os
import warnings

import torch

from .constants import RAY_REACH

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

ENABLED = os.environ.get("HEXFIELD_TRITON_RAYTAP_TRAIN", "1") == "1"

_RT_TRAIN_BM = int(os.environ.get("HEXFIELD_RT_TRAIN_BM", "32"))
_RT_TRAIN_BK = int(os.environ.get("HEXFIELD_RT_TRAIN_BK", "64"))
_RT_TRAIN_WARPS = int(os.environ.get("HEXFIELD_RT_TRAIN_WARPS", "4"))

def _ref_fwd(x, idx_taps, reach, alpha_full, O_full, P_full, corb):
    """Reference forward: [center ; naive taps] (identical numerics)."""
    from ._raytap import ray_tap_taps_naive

    if O_full.numel():
        taps = ray_tap_taps_naive(
            x, idx_taps, reach, alpha_full, O_full, P_full, corb
        )
    else:
        taps = ray_tap_taps_naive(x, idx_taps, reach, alpha_full, corb)
    return torch.cat([x.unsqueeze(2), taps], dim=2)


def _ref_bwd(x, idx_taps, reach, alpha_full, O_full, P_full, corb, grad_out,
             need_x, need_alpha, need_tables):
    """Reference backward (the K2 eager math on the 7-slot grad layout)."""
    from ._raytap import _effective_alpha, _masked_gather, _tap_flat_index, _tap_vis

    b, n, c = x.shape
    half = corb // 2
    g = c // corb
    has_lut = bool(O_full.numel())
    reach_l = reach.to(torch.long)
    x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
    O_arg = O_full if has_lut else None
    P_arg = P_full if has_lut else None
    grad_x = grad_out[:, :, 0].contiguous() if need_x else None
    grad_x_ext = None
    if need_x:
        grad_x_ext = torch.zeros_like(x_ext)
    grad_alpha = torch.zeros_like(alpha_full) if need_alpha else None
    grad_O = torch.zeros_like(O_full) if (need_tables and has_lut) else None
    grad_P = torch.zeros_like(P_full) if (need_tables and has_lut) else None
    for d in range(6):
        go_d = grad_out[:, :, 1 + d]
        if need_alpha or (need_tables and has_lut):
            xgv = _masked_gather(x_ext, idx_taps, reach_l, corb, d)
            contrib = xgv * go_d.unsqueeze(2)
            if need_alpha:
                grad_alpha += contrib.sum(dim=(0, 1))
            if need_tables and has_lut:
                contrib_flat = contrib.reshape(b * n, RAY_REACH, c)
                own_idx = (
                    reach_l[:, :, 0, d].reshape(b * n, 1, 1).expand(-1, RAY_REACH, c)
                )
                grad_O.scatter_add_(0, own_idx, contrib_flat)
                opp_idx = (
                    reach_l[:, :, 1, d].reshape(b * n, 1, 1).expand(-1, RAY_REACH, c)
                )
                grad_P.scatter_add_(0, opp_idx, contrib_flat)
            del xgv
        if need_x:
            a_d = _effective_alpha(alpha_full, O_arg, P_arg, reach_l, d)
            vis = _tap_vis(reach_l, d)
            gxg = (
                (a_d * go_d.unsqueeze(2)).view(b, n, RAY_REACH, g, 2, half)
                * vis.view(b, n, RAY_REACH, 1, 2, 1).to(x.dtype)
            ).reshape(b, n * RAY_REACH, c)
            grad_x_ext.scatter_add_(1, _tap_flat_index(idx_taps, d, c), gxg)
    if need_x:
        grad_x = grad_x + grad_x_ext[:, :n]
    return grad_x, grad_alpha, grad_O, grad_P


if HAVE_TRITON:

    _FWD_FAILED: set = set()
    _BWD_FAILED: set = set()
    _TRITON_VER = getattr(triton, "__version__", "?")

    def _mark_failed(failed: set, kernel: str, key, err: Exception) -> None:
        failed.add(key)
        warnings.warn(
            f"hexfield: triton {kernel} failed for key={key} "
            f"({type(err).__name__}) under triton {_TRITON_VER}; using the "
            "eager reference path for this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    @triton.jit
    def _rt7_train_fwd_kernel(
        x_ptr, idx_ptr, reach_ptr, alpha_ptr, O_ptr, P_ptr, out_ptr,
        B, Npad, C,
        IS_FP16: tl.constexpr, CORB: tl.constexpr, HAS_LUT: tl.constexpr,
        BM: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_c = tl.program_id(1)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        c_offs = pid_c * BK + tl.arange(0, BK)
        c_ok = c_offs < C
        HALF: tl.constexpr = CORB // 2
        side_c = (c_offs % CORB) >= HALF  # (BK,) opp-half channels

        out_row = out_ptr + m_offs[:, None] * (7 * C) + c_offs[None, :]

        # Tap 0 (center) = the query row itself, stored in x.dtype as-is.
        ctr = tl.load(
            x_ptr + (m_offs * C)[:, None] + c_offs[None, :],
            mask=m_valid[:, None] & c_ok[None, :],
            other=0.0,
        )
        tl.store(out_row, ctr, mask=m_valid[:, None] & c_ok[None, :])

        for t in tl.static_range(6):
            rl_own = tl.load(
                reach_ptr + m_offs * 12 + 0 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            rl_opp = tl.load(
                reach_ptr + m_offs * 12 + 1 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            acc = tl.zeros((BM, BK), dtype=tl.float32)
            for k in tl.static_range(5):
                idx = tl.load(
                    idx_ptr + m_offs * 30 + t * 5 + k, mask=m_valid, other=Npad
                ).to(tl.int32)
                present = m_valid & (idx < Npad)
                vo = rl_own >= (k + 1)
                vp = rl_opp >= (k + 1)
                a = tl.load(
                    x_ptr + ((b_ids * Npad + idx) * C)[:, None] + c_offs[None, :],
                    mask=(present & (vo | vp))[:, None] & c_ok[None, :],
                    other=0.0,
                ).to(tl.float32)
                coeff = tl.load(
                    alpha_ptr + k * C + c_offs, mask=c_ok, other=0.0
                ).to(tl.float32)[None, :] + tl.zeros((BM, BK), dtype=tl.float32)
                if HAS_LUT:
                    for r in tl.static_range(6):
                        o_r = tl.load(
                            O_ptr + (r * 5 + k) * C + c_offs, mask=c_ok, other=0.0
                        ).to(tl.float32)
                        coeff += tl.where(rl_own[:, None] == r, o_r[None, :], 0.0)
                        p_r = tl.load(
                            P_ptr + (r * 5 + k) * C + c_offs, mask=c_ok, other=0.0
                        ).to(tl.float32)
                        coeff += tl.where(rl_opp[:, None] == r, p_r[None, :], 0.0)
                vis = tl.where(
                    side_c[None, :],
                    vp.to(tl.float32)[:, None],
                    vo.to(tl.float32)[:, None],
                )
                acc += (a * vis) * coeff
            res = acc.to(tl.float16) if IS_FP16 else acc
            tl.store(
                out_row + (1 + t) * C,
                res,
                mask=m_valid[:, None] & c_ok[None, :],
            )

    @triton.jit
    def _rt7_train_bwd_kernel(
        x_ptr, go_ptr, idx_ptr, reach_ptr, alpha_ptr, O_ptr, P_ptr,
        gx_ptr, ga_ptr, gO_ptr, gP_ptr,
        B, Npad, C,
        CORB: tl.constexpr, HAS_LUT: tl.constexpr,
        BM: tl.constexpr, BK: tl.constexpr,
    ):
        """One-pass backward, scatter-formulated exactly like the eager K2
        reference: each program owns a tile of QUERY rows i, re-gathers the
        forward's masked x rows, and

          * accumulates the table grads (contrib = xg * vis * go) bucketed by
            the row's reach state via masked-partial atomics (LUT mode; the
            no-LUT mode accumulates grad_alpha directly), and
          * atomically scatters gxc = coeff * vis * go into the tap SOURCE
            rows of gx (pre-seeded with the center passthrough by the
            wrapper).

        Every load is coalesced except the x-row gather (the forward's own
        access pattern); the gather-formulated grad_x variant was 40-100x
        slower here — its per-(t, k) serial chain (idx gather -> reach gather
        -> table select) defeats latency hiding, while this form keeps reach
        loads coalesced and hoisted per direction."""
        pid_m = tl.program_id(0)
        pid_c = tl.program_id(1)
        m_offs = pid_m * BM + tl.arange(0, BM)
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        c_offs = pid_c * BK + tl.arange(0, BK)
        c_ok = c_offs < C
        HALF: tl.constexpr = CORB // 2
        side_c = (c_offs % CORB) >= HALF

        for t in tl.static_range(6):
            rl_own = tl.load(
                reach_ptr + m_offs * 12 + 0 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            rl_opp = tl.load(
                reach_ptr + m_offs * 12 + 1 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            go = tl.load(
                go_ptr + (m_offs * 7 + 1 + t)[:, None] * C + c_offs[None, :],
                mask=m_valid[:, None] & c_ok[None, :],
                other=0.0,
            ).to(tl.float32)
            for k in tl.static_range(5):
                idx = tl.load(
                    idx_ptr + m_offs * 30 + t * 5 + k, mask=m_valid, other=Npad
                ).to(tl.int32)
                present = m_valid & (idx < Npad)
                vo = rl_own >= (k + 1)
                vp = rl_opp >= (k + 1)
                live = present & (vo | vp)
                vis = tl.where(
                    side_c[None, :],
                    vp.to(tl.float32)[:, None],
                    vo.to(tl.float32)[:, None],
                )
                xg = tl.load(
                    x_ptr + ((b_ids * Npad + idx) * C)[:, None] + c_offs[None, :],
                    mask=live[:, None] & c_ok[None, :],
                    other=0.0,
                ).to(tl.float32)
                contrib = (xg * vis) * go
                coeff = tl.load(
                    alpha_ptr + k * C + c_offs, mask=c_ok, other=0.0
                ).to(tl.float32)[None, :] + tl.zeros((BM, BK), dtype=tl.float32)
                if HAS_LUT:
                    for r in tl.static_range(6):
                        own_r = rl_own == r
                        opp_r = rl_opp == r
                        s_o = tl.sum(
                            tl.where(own_r[:, None] & c_ok[None, :], contrib, 0.0),
                            axis=0,
                        )
                        tl.atomic_add(
                            gO_ptr + (r * 5 + k) * C + c_offs, s_o, mask=c_ok
                        )
                        s_p = tl.sum(
                            tl.where(opp_r[:, None] & c_ok[None, :], contrib, 0.0),
                            axis=0,
                        )
                        tl.atomic_add(
                            gP_ptr + (r * 5 + k) * C + c_offs, s_p, mask=c_ok
                        )
                        o_r = tl.load(
                            O_ptr + (r * 5 + k) * C + c_offs, mask=c_ok, other=0.0
                        ).to(tl.float32)
                        coeff += tl.where(own_r[:, None], o_r[None, :], 0.0)
                        p_r = tl.load(
                            P_ptr + (r * 5 + k) * C + c_offs, mask=c_ok, other=0.0
                        ).to(tl.float32)
                        coeff += tl.where(opp_r[:, None], p_r[None, :], 0.0)
                else:
                    s_a = tl.sum(
                        tl.where(c_ok[None, :], contrib, 0.0), axis=0
                    )
                    tl.atomic_add(ga_ptr + k * C + c_offs, s_a, mask=c_ok)
                gxc = (coeff * go) * vis
                tl.atomic_add(
                    gx_ptr + ((b_ids * Npad + idx) * C)[:, None] + c_offs[None, :],
                    gxc,
                    mask=live[:, None] & c_ok[None, :],
                )

    def _launch_grid(b: int, npad: int, c: int):
        rows = b * npad
        return (
            triton.cdiv(rows, _RT_TRAIN_BM),
            triton.cdiv(c, _RT_TRAIN_BK),
        )

    @torch.library.custom_op("hexfield_eq::rt_taps7_train_fwd", mutates_args=())
    def _rt_taps7_train_fwd(
        x: torch.Tensor,
        idx_taps: torch.Tensor,
        reach: torch.Tensor,
        alpha_full: torch.Tensor,
        O_full: torch.Tensor,
        P_full: torch.Tensor,
        corb: int,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        has_lut = bool(O_full.numel())
        key = (c, has_lut)
        if key not in _FWD_FAILED:
            try:
                xc = x.contiguous()
                idxc = idx_taps.contiguous()
                rch = reach.contiguous()
                a32 = alpha_full.to(torch.float32).contiguous()
                if has_lut:
                    o32 = O_full.to(torch.float32).contiguous()
                    p32 = P_full.to(torch.float32).contiguous()
                else:
                    o32 = p32 = a32
                out = torch.empty(
                    (b, npad, 7, c), dtype=x.dtype, device=x.device
                )
                _rt7_train_fwd_kernel[_launch_grid(b, npad, c)](
                    xc, idxc, rch, a32, o32, p32, out,
                    b, npad, c,
                    IS_FP16=(x.dtype == torch.float16),
                    CORB=corb, HAS_LUT=has_lut,
                    BM=_RT_TRAIN_BM, BK=_RT_TRAIN_BK,
                    num_warps=_RT_TRAIN_WARPS,
                )
                return out
            except Exception as err:
                _mark_failed(_FWD_FAILED, "rt_taps7_train_fwd", key, err)
        with torch.no_grad():
            return _ref_fwd(x, idx_taps, reach, alpha_full, O_full, P_full, corb)

    @_rt_taps7_train_fwd.register_fake
    def _rt_taps7_train_fwd_fake(
        x, idx_taps, reach, alpha_full, O_full, P_full, corb
    ):
        return x.new_empty((x.shape[0], x.shape[1], 7, x.shape[2]))

    @torch.library.custom_op("hexfield_eq::rt_taps7_train_bwd", mutates_args=())
    def _rt_taps7_train_bwd(
        x: torch.Tensor,
        grad_out: torch.Tensor,
        idx_taps: torch.Tensor,
        reach: torch.Tensor,
        alpha_full: torch.Tensor,
        O_full: torch.Tensor,
        P_full: torch.Tensor,
        corb: int,
    ) -> list[torch.Tensor]:
        """One-pass backward: [grad_x, grad_alpha, grad_O, grad_P] (the table
        grads fp32; zeros-shaped placeholders in the no-LUT mode). grad_x is
        seeded with the center passthrough and receives the direction terms by
        atomic scatter, exactly the eager K2 formulation."""

        b, npad, c = x.shape
        has_lut = bool(O_full.numel())
        key = (c, has_lut)
        dev = x.device
        if key not in _BWD_FAILED:
            try:
                xc = x.contiguous()
                goc = grad_out.contiguous()
                idxc = idx_taps.contiguous()
                rch = reach.contiguous()
                a32 = alpha_full.to(torch.float32).contiguous()
                if has_lut:
                    o32 = O_full.to(torch.float32).contiguous()
                    p32 = P_full.to(torch.float32).contiguous()
                else:
                    o32 = p32 = a32
                gx = goc[:, :, 0].contiguous()  # strided slice -> fresh buffer
                ga = torch.zeros((RAY_REACH, c), dtype=torch.float32, device=dev)
                gO = torch.zeros(
                    (RAY_REACH + 1, RAY_REACH, c), dtype=torch.float32, device=dev
                )
                gP = torch.zeros_like(gO)
                _rt7_train_bwd_kernel[_launch_grid(b, npad, c)](
                    xc, goc, idxc, rch, a32, o32, p32,
                    gx, ga, gO, gP,
                    b, npad, c,
                    CORB=corb, HAS_LUT=has_lut,
                    BM=_RT_TRAIN_BM, BK=_RT_TRAIN_BK,
                    num_warps=_RT_TRAIN_WARPS,
                )
                if has_lut:
                    # Every row has exactly one own-reach state, so the O
                    # buckets partition the full contrib sum == grad_alpha.
                    ga = gO.sum(dim=0)
                return [gx, ga, gO, gP]
            except Exception as err:
                _mark_failed(_BWD_FAILED, "rt_taps7_train_bwd", key, err)
        with torch.no_grad():
            gx, ga, gO, gP = _ref_bwd(
                x, idx_taps, reach, alpha_full, O_full, P_full, corb,
                grad_out, True, True, has_lut,
            )
            if gO is None:
                gO = torch.zeros(
                    (RAY_REACH + 1, RAY_REACH, c), dtype=torch.float32, device=dev
                )
                gP = torch.zeros_like(gO)
            return [gx, ga.float(), gO.float(), gP.float()]

    @_rt_taps7_train_bwd.register_fake
    def _rt_taps7_train_bwd_fake(
        x, grad_out, idx_taps, reach, alpha_full, O_full, P_full, corb
    ):
        c = x.shape[2]
        return [
            x.new_empty(x.shape),
            x.new_empty((RAY_REACH, c), dtype=torch.float32),
            x.new_empty((RAY_REACH + 1, RAY_REACH, c), dtype=torch.float32),
            x.new_empty((RAY_REACH + 1, RAY_REACH, c), dtype=torch.float32),
        ]

    def _setup_context(ctx, inputs, output):
        x, idx_taps, reach, alpha_full, O_full, P_full, corb = inputs
        ctx.save_for_backward(x, idx_taps, reach, alpha_full, O_full, P_full)
        ctx.corb = corb

    def _backward(ctx, grad_out):
        x, idx_taps, reach, alpha_full, O_full, P_full = ctx.saved_tensors
        corb = ctx.corb
        has_lut = bool(O_full.numel())
        gx, ga, gO, gP = _rt_taps7_train_bwd(
            x, grad_out, idx_taps, reach, alpha_full, O_full, P_full, corb
        )
        grad_alpha = ga.to(alpha_full.dtype) if ctx.needs_input_grad[3] else None
        grad_O = (
            gO.to(O_full.dtype)
            if (has_lut and ctx.needs_input_grad[4])
            else None
        )
        grad_P = (
            gP.to(P_full.dtype)
            if (has_lut and ctx.needs_input_grad[5])
            else None
        )
        grad_x = gx if ctx.needs_input_grad[0] else None
        return grad_x, None, None, grad_alpha, grad_O, grad_P, None

    _rt_taps7_train_fwd.register_autograd(_backward, setup_context=_setup_context)

    def rt_taps7_train(
        x: torch.Tensor,
        idx_taps: torch.Tensor,
        reach: torch.Tensor,
        alpha_full: torch.Tensor,
        O_full: torch.Tensor | None,
        P_full: torch.Tensor | None,
        corb: int,
    ) -> torch.Tensor:
        """(B, N, 7, C) tapped conv input [center ; 6 direction taps] with full
        autograd - the fused train-stream entry point. O_full/P_full None (or
        empty) selects the alpha-only aggregation."""

        if O_full is None:
            O_full = alpha_full.new_empty(0)
            P_full = alpha_full.new_empty(0)
        return _rt_taps7_train_fwd(
            x, idx_taps, reach, alpha_full, O_full, P_full, int(corb)
        )

else:  # pragma: no cover
    rt_taps7_train = None
