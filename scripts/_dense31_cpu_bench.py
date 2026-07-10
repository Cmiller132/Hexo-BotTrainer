#!/usr/bin/env python3
"""CPU micro-benchmark for one equipped ray-tap or dense31 convolution."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

from hexfield_eq import _dense31 as D31
from hexfield_eq import _raytap as RT


assert not torch.cuda.is_available(), "dense31 CPU bench must run CPU-only"
torch.set_num_threads(8)


def _inputs(batch: int, n: int = 448, c: int = 192):
    gen = torch.Generator().manual_seed(31 + batch)
    x = torch.randn(batch, n, c, generator=gen)
    idx = torch.randint(0, n + 1, (batch, n, 6, 5), generator=gen)
    reach = torch.randint(0, 6, (batch, n, 2, 6), generator=gen).to(torch.uint8)
    mask = torch.rand(batch, n, generator=gen) > 0.04
    w7 = torch.randn(7, c, c, generator=gen) / c**0.5
    w31 = torch.randn(31, c, c, generator=gen) / c**0.5
    bias = torch.randn(c, generator=gen) * 0.02
    alpha = torch.randn(5, 16, generator=gen) * 0.2
    alpha[0] += 1.0
    return x, idx, reach, mask, w7, w31, bias, alpha


def _ray(x, idx, reach, mask, weight, bias, alpha, function: bool):
    alpha_full = alpha.repeat(1, x.shape[-1] // alpha.shape[-1])
    tap_fn = RT.ray_tap_taps if function else RT.ray_tap_taps_naive
    taps = tap_fn(x, idx, reach, alpha_full, alpha.shape[-1])
    gathered = torch.cat([x.unsqueeze(2), taps], dim=2).reshape(x.shape[0], x.shape[1], -1)
    return (gathered @ weight.reshape(7 * x.shape[-1], weight.shape[-1]) + bias) * mask.unsqueeze(-1)


def _call(kind, path, values):
    x, idx, reach, mask, w7, w31, bias, alpha = values
    if kind == "raytap":
        return _ray(x, idx, reach, mask, w7, bias, alpha, path == "function")
    fn = D31.dense31_conv if path == "function" else D31.dense31_conv_naive
    return fn(x, idx, reach, w31, bias, mask, 16)


def _measure(kind, path, phase, base_values, warmup, iters):
    samples = []
    for rep in range(warmup + iters):
        x, idx, reach, mask, w7, w31, bias, alpha = base_values
        if phase == "fwd+bwd":
            x = x.detach().requires_grad_(True)
            w7 = w7.detach().requires_grad_(True)
            w31 = w31.detach().requires_grad_(True)
            bias = bias.detach().requires_grad_(True)
            alpha = alpha.detach().requires_grad_(True)
        values = (x, idx, reach, mask, w7, w31, bias, alpha)
        gc.collect()
        started = time.perf_counter()
        if phase == "fwd":
            with torch.no_grad():
                out = _call(kind, path, values)
        else:
            out = _call(kind, path, values)
            out.square().mean().backward()
        elapsed = (time.perf_counter() - started) * 1000.0
        if rep >= warmup:
            samples.append(elapsed)
        del out, values
    return statistics.median(samples), min(samples), max(samples)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("docs/DENSE31_CPU_BENCH.csv"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = []
    for batch in (2, 8):
        values = _inputs(batch)
        for phase in ("fwd", "fwd+bwd"):
            for kind in ("raytap", "dense31"):
                for path in ("reference", "function"):
                    median, low, high = _measure(
                        kind, path, phase, values, args.warmup, args.iters
                    )
                    row = dict(batch=batch, npad=448, channels=192, phase=phase,
                               operator=kind, path=path, median_ms=median,
                               min_ms=low, max_ms=high)
                    rows.append(row)
                    print(
                        f"B={batch} {phase:7s} {kind:7s} {path:9s} "
                        f"median={median:9.3f} ms",
                        flush=True,
                    )
        del values
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
