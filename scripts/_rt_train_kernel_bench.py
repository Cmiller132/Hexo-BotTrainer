"""Microbench the three _triton_raytap_train kernels at live shape + variants."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

sys.argv = sys.argv  # noqa

from hexfield_eq import _raytap as RT
from hexfield_eq import _triton_raytap_train as RTT
from hexfield_eq.constants import DIRECTIONS, RAY_REACH, RAYLEN_SLOTS

DEV = torch.device("cuda")
B, N, NPAD, C, CORB = 59, 470, 512, 192, 16


def _hexdist(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


_DISK = [(dq, dr) for dq in range(-4, 5) for dr in range(-4, 5) if _hexdist(dq, dr) <= 4]


def build():
    rng = random.Random(0)
    nprng = np.random.default_rng(0)
    coords = torch.zeros(B, NPAD, 2, dtype=torch.long)
    mask = torch.zeros(B, NPAD, dtype=torch.bool)
    raylen = torch.zeros(B, NPAD, RAYLEN_SLOTS, dtype=torch.uint8)
    stones = [(0, 0)]
    support = set(_DISK)
    while len(support) < N:
        q, r = stones[rng.randrange(len(stones))]
        dq, dr = DIRECTIONS[rng.randrange(6)]
        s = (q + dq, r + dr)
        if s in stones:
            continue
        stones.append(s)
        for dq2, dr2 in _DISK:
            support.add((s[0] + dq2, s[1] + dr2))
    cells = sorted(support)[:N]
    for i in range(B):
        coords[i, :N] = torch.tensor(cells, dtype=torch.long)
        mask[i, :N] = True
        raylen[i, :N] = torch.from_numpy(
            nprng.integers(0, RAY_REACH + 1, size=(N, RAYLEN_SLOTS)).astype(np.uint8)
        )
    coords, mask, raylen = coords.to(DEV), mask.to(DEV), raylen.to(DEV)
    idx_taps = RT.build_tap_gather_index(coords, mask)
    reach = RT.build_tap_reach(raylen)
    x = torch.randn(B, NPAD, C, device=DEV)
    alpha = torch.randn(RAY_REACH, C, device=DEV)
    O = torch.randn(RAY_REACH + 1, RAY_REACH, C, device=DEV)
    P = torch.randn(RAY_REACH + 1, RAY_REACH, C, device=DEV)
    go = torch.randn(B, NPAD, 7, C, device=DEV)
    return x, idx_taps, reach, mask, alpha, O, P, go


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    x, idx_taps, reach, mask, alpha, O, P, go = build()
    empty = alpha.new_empty(0)

    t = bench(lambda: RTT._rt_taps7_train_fwd(x, idx_taps, reach, alpha, O, P, CORB))
    print(f"fwd lut  : {t:7.2f} ms")
    t = bench(lambda: RTT._rt_taps7_train_fwd(x, idx_taps, reach, alpha, empty, empty, CORB))
    print(f"fwd alpha: {t:7.2f} ms")
    t = bench(lambda: RTT._rt_taps7_train_bwd(x, go, idx_taps, reach, alpha, O, P, CORB))
    print(f"bwd lut  : {t:7.2f} ms")
    t = bench(lambda: RTT._rt_taps7_train_bwd(x, go, idx_taps, reach, alpha, empty, empty, CORB))
    print(f"bwd alpha: {t:7.2f} ms")


if __name__ == "__main__":
    main()
