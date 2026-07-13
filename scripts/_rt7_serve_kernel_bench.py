"""hex_ray_taps7 serve-kernel A/B: lut (additive) vs none, fp16 serve shapes."""

from __future__ import annotations

import random

import numpy as np
import torch

from hexfield_eq import _raytap as RT
from hexfield_eq._triton_conv import hex_ray_taps7
from hexfield_eq.constants import DIRECTIONS, RAY_REACH, RAYLEN_SLOTS
from hexfield_eq._triton_ray import build_ray_gather_index

DEV = torch.device("cuda")
B, N, NPAD, C, CORB = 260, 470, 512, 192, 16


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
        # Serve-like truncation mix: mostly short reaches.
        raylen[i, :N] = torch.from_numpy(
            np.minimum(
                nprng.geometric(0.45, size=(N, RAYLEN_SLOTS)) - 1, RAY_REACH
            ).astype(np.uint8)
        )
    coords, mask, raylen = coords.to(DEV), mask.to(DEV), raylen.to(DEV)
    ray_idx = build_ray_gather_index(coords, mask)
    reach = RT.build_tap_reach(raylen)
    x = torch.randn(B, NPAD, C, device=DEV, dtype=torch.float16)
    alpha = (torch.randn(RAY_REACH, C, device=DEV) * 0.3).to(torch.float16)
    O = (torch.randn(RAY_REACH + 1, RAY_REACH, C, device=DEV) * 0.1).to(torch.float16)
    P = (torch.randn(RAY_REACH + 1, RAY_REACH, C, device=DEV) * 0.1).to(torch.float16)
    return x, ray_idx, reach, alpha, O, P


def bench(fn, iters=50):
    for _ in range(5):
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
    x, ray_idx, reach, alpha, O, P = build()
    with torch.no_grad():
        t_none = bench(lambda: hex_ray_taps7(x, ray_idx, reach, alpha, CORB))
        t_lut = bench(lambda: hex_ray_taps7(x, ray_idx, reach, alpha, CORB, O, P))
    print(f"taps7 none: {t_none:6.3f} ms")
    print(f"taps7 lut : {t_lut:6.3f} ms  (tax {100*(t_lut/t_none-1):+.1f}%)")


if __name__ == "__main__":
    main()
