"""Fused Triton train-stream ray-tap gate (_triton_raytap_train).

Oracle: `_raytap.ray_tap_taps_naive` (the numerics reference for every ray-tap
path, T8's oracle) — forward values and ALL gradients (x, alpha, O, P) of the
fused custom op must match it on CUDA, both lut modes, at small and live-ish
shapes, including pad rows and sentinel taps.

The op's grad_x kernel is gather-formulated via the opposite-tap bijection, so
`idx_taps` MUST be geometric (built by `_raytap.build_tap_gather_index` from
coords/mask, as trunk() does) — these tests build it exactly that way.

CUDA-only (self-skips elsewhere); the CPU suite (test_hexfield_eq_raytap.py)
is untouched and keeps pinning the eager K2 island the op falls back to.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from hexfield_eq import _raytap as RT
from hexfield_eq.constants import DIRECTIONS, RAY_REACH, RAYLEN_SLOTS

if not torch.cuda.is_available():  # pragma: no cover - CPU CI
    pytest.skip("CUDA required for the fused train ray-tap gate", allow_module_level=True)

try:
    from hexfield_eq import _triton_raytap_train as RTT
except Exception:  # pragma: no cover
    RTT = None

if RTT is None or not RTT.HAVE_TRITON:  # pragma: no cover
    pytest.skip("triton unavailable", allow_module_level=True)

DEV = torch.device("cuda")


def _hexdist(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


_DISK = [
    (dq, dr)
    for dq in range(-4, 5)
    for dr in range(-4, 5)
    if _hexdist(dq, dr) <= 4
]


def _blob_inputs(b: int, n: int, npad: int, c: int, corb: int, seed: int,
                 lut: bool):
    """Geometric batch: random stone-blob supports (radius-4 disks), real tap
    gather index from coords/mask, random raylen wire. Pad rows carry raylen 0
    and mask False (the D-S13 convention)."""

    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    coords = torch.zeros(b, npad, 2, dtype=torch.long)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    raylen = torch.zeros(b, npad, RAYLEN_SLOTS, dtype=torch.uint8)
    for i in range(b):
        stones = [(0, 0)]
        support = set(_DISK)
        while len(support) < n:
            q, r = stones[rng.randrange(len(stones))]
            dq, dr = DIRECTIONS[rng.randrange(6)]
            s = (q + dq, r + dr)
            if s in stones:
                continue
            stones.append(s)
            for dq2, dr2 in _DISK:
                support.add((s[0] + dq2, s[1] + dr2))
        cells = sorted(support)[:n]
        coords[i, :n] = torch.tensor(cells, dtype=torch.long)
        mask[i, :n] = True
        raylen[i, :n] = torch.from_numpy(
            nprng.integers(0, RAY_REACH + 1, size=(n, RAYLEN_SLOTS)).astype(np.uint8)
        )
    coords = coords.to(DEV)
    mask = mask.to(DEV)
    raylen = raylen.to(DEV)
    idx_taps = RT.build_tap_gather_index(coords, mask)
    reach = RT.build_tap_reach(raylen)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(b, npad, c, generator=g).to(DEV)
    alpha = torch.randn(RAY_REACH, c, generator=g).to(DEV) * 0.5
    if lut:
        O = torch.randn(RAY_REACH + 1, RAY_REACH, c, generator=g).to(DEV) * 0.3
        P = torch.randn(RAY_REACH + 1, RAY_REACH, c, generator=g).to(DEV) * 0.3
    else:
        O = P = None
    return x, idx_taps, reach, mask, alpha, O, P, corb


def _run_pair(shape, lut: bool, seed: int = 0):
    """(fused out+grads, reference out+grads) on identical leaf tensors."""

    b, n, npad, c, corb = shape
    x0, idx_taps, reach, mask, alpha0, O0, P0, corb = _blob_inputs(
        b, n, npad, c, corb, seed, lut
    )
    results = []
    for use_fused in (True, False):
        x = x0.clone().requires_grad_(True)
        alpha = alpha0.clone().requires_grad_(True)
        O = O0.clone().requires_grad_(True) if lut else None
        P = P0.clone().requires_grad_(True) if lut else None
        if use_fused:
            out = RTT.rt_taps7_train(x, idx_taps, reach, alpha, O, P, corb)
        else:
            if lut:
                taps = RT.ray_tap_taps_naive(x, idx_taps, reach, alpha, O, P, corb)
            else:
                taps = RT.ray_tap_taps_naive(x, idx_taps, reach, alpha, corb)
            out = torch.cat([x.unsqueeze(2), taps], dim=2)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        w = torch.randn(out.shape, generator=g).to(DEV)
        (out * w).sum().backward()
        results.append(
            (
                out.detach(),
                x.grad.detach().clone(),
                alpha.grad.detach().clone(),
                O.grad.detach().clone() if lut else None,
                P.grad.detach().clone() if lut else None,
            )
        )
    return results


SMALL = (2, 49, 64, 32, 16)      # tiny support, one c-tile, pad tail
LIVE = (8, 470, 512, 192, 16)    # live main_3 microbucket geometry


@pytest.mark.parametrize("lut", (False, True), ids=("alpha", "lut2"))
@pytest.mark.parametrize("shape", (SMALL, LIVE), ids=("small", "live"))
def test_fused_matches_naive(shape, lut):
    (f_out, f_gx, f_ga, f_gO, f_gP), (r_out, r_gx, r_ga, r_gO, r_gP) = _run_pair(
        shape, lut
    )
    # Forward: same products in the same k order, fp32 accumulate.
    torch.testing.assert_close(f_out, r_out, rtol=1e-5, atol=1e-5)
    # grad_x: <= 31 fp32 terms per element (gather- vs scatter-ordered).
    torch.testing.assert_close(f_gx, r_gx, rtol=1e-4, atol=1e-4)
    # Table grads: reductions over B*N*6 terms, atomic order differs.
    tol = dict(rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(f_ga, r_ga, **tol)
    if lut:
        torch.testing.assert_close(f_gO, r_gO, **tol)
        torch.testing.assert_close(f_gP, r_gP, **tol)


def test_fused_kernel_actually_ran():
    """The Triton path must not have silently latched a failure fallback for
    the widths under test (else the parity above proved nothing)."""

    assert (192, True) not in RTT._FWD_FAILED
    assert (192, True) not in RTT._BWD_FAILED
    assert (32, False) not in RTT._FWD_FAILED
    assert (32, False) not in RTT._BWD_FAILED


def test_grad_x_zero_where_masked():
    """Pad rows have raylen 0 and sentinel taps everywhere: their direction-
    tap gradient contribution is exactly zero (only the center passthrough
    remains)."""

    b, n, npad, c, corb = SMALL
    x, idx_taps, reach, mask, alpha, O, P, corb = _blob_inputs(
        b, n, npad, c, corb, 7, True
    )
    x = x.requires_grad_(True)
    out = RTT.rt_taps7_train(x, idx_taps, reach, alpha, O, P, corb)
    go = torch.zeros_like(out)
    go[:, :, 1:] = 1.0  # direction taps only; center grad excluded
    out.backward(go)
    # Pad rows (mask False rows n..npad) never appear as a tap source: real
    # rows' gather indices point at support rows only, and pad rows' own taps
    # are sentinels. grad_x on pad rows must be exactly zero.
    assert x.grad[:, n:].abs().max().item() == 0.0


def test_compile_keeps_op_in_graph():
    """Under torch.compile the custom op must not force a graph break (the
    whole point of retiring the K2 eager island for the compiled trainer)."""

    b, n, npad, c, corb = SMALL
    x, idx_taps, reach, mask, alpha, O, P, corb = _blob_inputs(
        b, n, npad, c, corb, 11, True
    )

    def f(x, alpha, O, P):
        return RTT.rt_taps7_train(x, idx_taps, reach, alpha, O, P, corb).sum()

    explanation = torch._dynamo.explain(f)(x, alpha, O, P)
    assert explanation.graph_break_count == 0, explanation
    torch._dynamo.reset()
    compiled = torch.compile(f)
    x1 = x.clone().requires_grad_(True)
    loss = compiled(x1, alpha, O, P)
    loss.backward()
    x2 = x.clone().requires_grad_(True)
    ref = f(x2, alpha, O, P)
    ref.backward()
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-4)
