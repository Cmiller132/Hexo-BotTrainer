#!/usr/bin/env python
"""Benchmark the gathered ray-attention Triton kernel (spec D-S36/D-S37)
against the flex and materialized L-block serve paths.

IDLE-GPU ONLY: refuses to run if nvidia-smi reports > 10% GPU utilization
(a prefit/train ladder owns the box) unless --force is passed. Never run this
next to a live run — the materialized path alone transiently allocates
~0.7 GB at the largest default shape.

What is timed (per L block, fp16, no-grad, CUDA, median over --iters after
--warmup):
  * materialized : per-forward _build_ray_bias (the (B, 6, N, N) additive
                   bias + live-mask transient) + RayAttention (sdpa impl);
  * flex         : per-forward _build_ray_flex_bias carrier + RayAttention
                   through compiled flex_attention (first call per shape
                   compiles — covered by the warmup);
  * kernel       : per-forward slot-bias resolve + RayAttention through the
                   hexfield_eq::ray_attn Triton kernel with a PREBUILT gather
                   index; the (B, Npad, 32) index build is timed separately
                   (idx_ms) because a real forward builds it ONCE and shares
                   it across every L block in the layout.

Grid: --npad {128 256 512 768} x --batch {1 8 24 48} x blockers {on off}.
Boards are the Npad hex cells closest to the origin (unique axial coords,
mask all-live: the worst case for every path); raylen is uniform random 0..5.

Tile knobs ride the HEXFIELD_RAY_BM / HEXFIELD_RAY_WARPS env (read once at
import by _triton_ray.py), so a sweep looks like:

  for bm in 8 16 32; do HEXFIELD_RAY_BM=$bm python scripts/bench_eq_ray_kernel.py ...; done

Typical WSL invocation (hexgt-build venv, repo root):

  PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:\
packages/hexo_utils/python:packages/hexo_train/python \
  /root/.venvs/hexgt-build/bin/python scripts/bench_eq_ray_kernel.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys

# The kernel import in model.py is env-gated; set it BEFORE the package import
# so RayAttention routes _RayGatherBias carriers to the Triton op.
os.environ.setdefault("HEXFIELD_EQ_TRITON_RAY", "1")

import torch  # noqa: E402


def _gpu_util() -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    util, mem = out.strip().splitlines()[0].split(",")
    return int(util.strip()), int(mem.strip())


def _board(npad: int, batch: int, device: str):
    from hexfield_eq.geometry import disk_offsets, hex_dist

    radius = 1
    while len(disk_offsets(radius)) < npad:
        radius += 1
    cells = sorted(disk_offsets(radius), key=lambda c: (hex_dist(*c), c))[:npad]
    coords = torch.tensor(cells, dtype=torch.long, device=device)
    coords = coords.unsqueeze(0).expand(batch, npad, 2).contiguous()
    mask = torch.ones(batch, npad, dtype=torch.bool, device=device)
    raylen = torch.randint(
        0, 6, (batch, npad, 12), dtype=torch.uint8, device=device
    )
    return coords, mask, raylen


def _time_cuda(fn, warmup: int, iters: int) -> float:
    """Median wall ms of fn() via CUDA events."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npad", type=int, nargs="+", default=[128, 256, 512, 768])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 24, 48])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--blockers", choices=["on", "off", "both"], default="both",
        help="raylen blocker gating arms to bench (default both)",
    )
    ap.add_argument(
        "--skip", nargs="*", default=[],
        choices=["materialized", "flex", "kernel", "kernel2"],
        help="paths to skip (e.g. flex, whose per-shape compile is slow)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="run even if the GPU is > 10%% utilized (NEVER next to a prefit)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; nothing to bench.", file=sys.stderr)
        return 2
    try:
        util, mem = _gpu_util()
    except Exception as exc:  # pragma: no cover - no nvidia-smi
        print(f"nvidia-smi failed ({exc}); refusing without --force.", file=sys.stderr)
        if not args.force:
            return 2
        util, mem = -1, -1
    if util > 10 and not args.force:
        print(
            f"GPU busy ({util}% util, {mem} MiB used) — a run owns the box; "
            "re-run in an idle window (or --force if you are SURE).",
            file=sys.stderr,
        )
        return 2

    from hexfield_eq import _triton_ray as TR
    from hexfield_eq import constants as C
    from hexfield_eq import model as M
    from hexfield_eq._triton_ray import build_ray_gather_index, slot_bias_rows
    from hexfield_eq.model import HexfieldNet

    if M._ray_attn_fused is None:
        print("triton ray kernel unavailable (import failed)", file=sys.stderr)
        return 2

    dev = "cuda"
    torch.manual_seed(0)
    net = HexfieldNet(trunk_layout="CLA").eval()
    with torch.no_grad():
        tables = (
            net.bias_theta_l if C.GROUP_ORDER == 12 else net.ray_bias_free_tables
        )
        for p in tables:
            p.copy_(torch.randn_like(p) * 0.3)
    net = net.half().to(dev)
    attn = net.ray_blocks[0].attn
    rows = slot_bias_rows().to(dev)
    blocker_arms = {"on": [True], "off": [False], "both": [True, False]}[args.blockers]

    print(
        f"# gathered ray kernel bench | triton v1 BM={TR._BM} WARPS={TR._WARPS} "
        f"| v2 BM={TR._V2_BM} WARPS={TR._V2_WARPS} | C={C.CHANNELS} "
        f"head_dim={C.CHANNELS // C.RAY_HEADS} | iters={args.iters} warmup={args.warmup}"
    )
    hdr = (
        f"{'B':>3} {'Npad':>5} {'blk':>3} | {'mat_ms':>8} {'flex_ms':>8} "
        f"{'kern_ms':>8} {'kern2_ms':>8} {'idx_ms':>7} | {'mat/kern':>8} "
        f"{'flex/kern':>9} {'kern/kern2':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for npad in args.npad:
        for batch in args.batch:
            coords, mask, raylen = _board(npad, batch, dev)
            x = torch.randn(batch, npad, C.CHANNELS, device=dev).half()
            for blockers in blocker_arms:
                netb = net
                netb._ray_blockers = blockers
                rl = raylen if blockers else None
                results: dict[str, float | None] = {}

                def bench(name, fn):
                    if name in args.skip:
                        results[name] = None
                        return
                    try:
                        with torch.no_grad():
                            results[name] = _time_cuda(fn, args.warmup, args.iters)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        results[name] = float("nan")

                bench(
                    "materialized",
                    lambda: attn(x, netb._build_ray_bias(coords, mask, rl, 0)),
                )
                bench(
                    "flex",
                    lambda: attn(x, netb._build_ray_flex_bias(coords, mask, rl, 0)),
                )
                idx = build_ray_gather_index(coords, mask)
                seq = mask.sum(dim=1).to(torch.int32)  # all-live boards
                rl_arg = (
                    raylen if blockers
                    else torch.empty(0, dtype=torch.uint8, device=dev)
                )

                def kern_fwd():
                    slot_bias = netb._ray_bias_table(0)[rows].to(torch.float16)
                    carrier = M._RayGatherBias(idx, slot_bias, rl_arg, seq, blockers)
                    return attn(x, carrier)

                # v1 (program-per-(batch,head)) then v2 (all-heads-per-program):
                # flip the module global BETWEEN timed sections; ray_attn reads it
                # at call time, so `attn(x, carrier)` routes to the chosen kernel.
                TR._USE_V2 = False
                bench("kernel", kern_fwd)
                TR._USE_V2 = True
                bench("kernel2", kern_fwd)
                TR._USE_V2 = False
                results["idx"] = _time_cuda(
                    lambda: build_ray_gather_index(coords, mask),
                    args.warmup,
                    args.iters,
                )

                def fmt(v):
                    return "  skip" if v is None else f"{v:8.3f}"

                def ratio(a, b):
                    if a is None or b is None or b != b or a != a or b == 0:
                        return "     n/a"
                    return f"{a / b:8.2f}x"

                km = results["kernel"]
                km2 = results["kernel2"]
                print(
                    f"{batch:>3} {npad:>5} {'on' if blockers else 'off':>3} | "
                    f"{fmt(results['materialized'])} {fmt(results['flex'])} "
                    f"{fmt(km)} {fmt(km2)} {results['idx']:7.3f} | "
                    f"{ratio(results['materialized'], km)} {ratio(results['flex'], km)} "
                    f"{ratio(km, km2)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
