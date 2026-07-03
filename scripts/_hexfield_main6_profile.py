"""Forward breakdown at main_6 shapes (c=128, radius-4 blob supports).

Answers: where does the serve forward spend time at the LIVE run's shapes
(N ~ 230-680, groups of ~100-250 rows) — conv gather vs GEMM vs flex vs rest.

Run (GPU free, WSL):
  HEXFIELD_CHANNELS=128 HEXFIELD_SERVE_FLEX=1 \
  PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python \
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_main6_profile.py [compiled]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("HEXFIELD_CHANNELS", "128")
os.environ.setdefault("HEXFIELD_SERVE_FLEX", "1")

import numpy as np
import torch

from hexfield.constants import DIRECTIONS, NUM_FEATURES, NUM_TOKENS
from hexfield.model import HexfieldNet

QUANT = 64
PAIR_CEILING = 3.8e7
DISK_R = 4  # HEXFIELD_SUPPORT_RADIUS in the live run


def hexdist(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


DISK = [
    (dq, dr)
    for dq in range(-DISK_R, DISK_R + 1)
    for dr in range(-DISK_R, DISK_R + 1)
    if hexdist(dq, dr) <= DISK_R
]


def make_blob(n_target: int, rng: np.random.Generator):
    """Random adjacent stone walk; support = union of radius-4 disks. Returns
    (coords (N,2) int64, nbr (N,6) row-local with missing -> N)."""

    stones = [(0, 0)]
    support: set[tuple[int, int]] = set(DISK)
    while len(support) < n_target:
        base = stones[rng.integers(len(stones))]
        d = DIRECTIONS[rng.integers(6)]
        s = (base[0] + d[0], base[1] + d[1])
        if s in [tuple(x) for x in stones]:
            continue
        stones.append(s)
        for dq, dr in DISK:
            support.add((s[0] + dq, s[1] + dr))
    cells = sorted(support)[: n_target]  # trim to exact N for shape control
    cellset = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    coords = np.array(cells, dtype=np.int64)
    nbr = np.full((n, 6), n, dtype=np.int64)
    for i, (q, r) in enumerate(cells):
        for t, (dq, dr) in enumerate(DIRECTIONS):
            j = cellset.get((q + dq, r + dr))
            if j is not None:
                nbr[i, t] = j
    return coords, nbr


def make_batch(b: int, n: int, device, rng):
    npad = ((n + QUANT - 1) // QUANT) * QUANT
    feats = torch.zeros(b, npad, NUM_FEATURES, dtype=torch.float16)
    nbrs = torch.full((b, npad, 6), npad, dtype=torch.long)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    coords = torch.zeros(b, npad, 2, dtype=torch.long)
    for k in range(4):  # 4 distinct blobs cycled over the batch
        c, nb = make_blob(n, rng)
        rows = range(k, b, 4)
        for i in rows:
            feats[i, :n] = torch.from_numpy(
                rng.standard_normal((n, NUM_FEATURES)).astype(np.float16)
            )
            nbrs[i, :n] = torch.from_numpy(nb)
            # pad-row nbr already npad; blob 'missing -> n' must become npad
            nbrs[i, :n][nbrs[i, :n] == n] = npad
            mask[i, :n] = True
            coords[i, :n] = torch.from_numpy(c)
    return (
        feats.to(device),
        nbrs.to(device),
        mask.to(device),
        coords.to(device),
    )


def timeit(fn, warmup=8, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def categorize(name: str) -> str:
    n = name.lower()
    if "flex" in n:
        return "flex-attn"
    if "gather" in n or "index" in n or "scatter" in n:
        return "gather/index"
    if any(s in n for s in ("gemm", "cutlass", "nvjet", "addmm", "matmul", "mm_", "sgemm", "hgemm", "wgmma", "conv")):
        return "gemm"
    if "softmax" in n:
        return "softmax"
    if "norm" in n:
        return "layernorm"
    if "triton" in n:
        return "triton-fused"
    if any(s in n for s in ("elementwise", "vectorized", "reduce", "cat", "copy", "fill", "where", "clamp")):
        return "elementwise/copy"
    return "other"


HALF = "half" in sys.argv[1:]


def profile_forward(fwd, args, tag):
    from torch.profiler import ProfilerActivity, profile

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not HALF):
            fwd(*args)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(10):
            run()
    torch.cuda.synchronize()
    agg: dict[str, float] = {}
    rows = []
    for evt in prof.key_averages():
        if evt.self_device_time_total <= 0:
            continue
        cat = categorize(evt.key)
        agg[cat] = agg.get(cat, 0.0) + evt.self_device_time_total
    total = sum(agg.values())
    print(f"\n  [{tag}] kernel-category breakdown (10 fwd):")
    for cat, us in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:16s} {us/1e3/10:7.2f} ms/fwd  {100*us/total:5.1f}%")
    top = sorted(
        (e for e in prof.key_averages() if e.self_device_time_total > 0),
        key=lambda e: -e.self_device_time_total,
    )[:12]
    print(f"  [{tag}] top kernels:")
    for e in top:
        print(f"    {e.self_device_time_total/1e3/10:7.2f} ms/fwd  {e.key[:110]}")


def conv_microbench(model, args):
    """Split one HexNodeConv at trunk width: gather vs GEMM vs mask."""
    feats, nbrs, mask, coords = args
    b, npad, _ = feats.shape
    c = model.stem.out_channels
    conv = model.conv_blocks[0].conv1
    x32 = torch.randn(b, npad, c, device=feats.device)  # LN output dtype (fp32)
    x16 = x32.half()
    self_idx = torch.arange(npad, device=feats.device).reshape(1, npad, 1).expand(b, -1, -1)
    gidx = torch.cat([self_idx, nbrs], dim=2)
    w = conv.weight.reshape(7 * c, c)

    def gather_only(x):
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        flat = gidx.reshape(b, npad * 7, 1).expand(-1, -1, c)
        return x_ext.gather(1, flat).reshape(b, npad, 7 * c)

    g32 = gather_only(x32)
    g16 = gather_only(x16)
    with torch.autocast("cuda", dtype=torch.float16):
        t_full32 = timeit(lambda: conv(x32, gidx, mask))
        t_full16 = timeit(lambda: conv(x16, gidx, mask))
        t_gather32 = timeit(lambda: gather_only(x32))
        t_gather16 = timeit(lambda: gather_only(x16))
        t_gemm16 = timeit(lambda: g16.half() @ w.half())
    t_gemm32 = timeit(lambda: g32 @ w)
    print(
        f"  conv micro (B={b}, Npad={npad}, C={c}):  full fp32-in {t_full32:.3f} / fp16-in {t_full16:.3f} ms"
        f"  | gather fp32 {t_gather32:.3f} / fp16 {t_gather16:.3f} ms  | gemm fp16 {t_gemm16:.3f} / fp32 {t_gemm32:.3f} ms"
    )


def main() -> None:
    compiled = "compiled" in sys.argv[1:]
    device = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = HexfieldNet().to(device).eval()
    if HALF:
        model = model.half()
    print(
        f"torch {torch.__version__}  channels {model.stem.out_channels}  "
        f"serve_flex {model._serve_flex}  flex_pair {model._flex_pair}  "
        f"half {HALF}  mode {'compiled' if compiled else 'eager'}"
    )

    fwd = model.forward_policy_value
    if compiled:
        fwd = torch.compile(model.forward_policy_value, dynamic=True)

    for n in (256, 384, 512, 640):
        s = ((n + QUANT - 1) // QUANT) * QUANT + NUM_TOKENS
        b = max(2, min(256, int(PAIR_CEILING / (s * s))))
        args = make_batch(b, n, device, rng)

        def run():
            if compiled:
                for t in args:
                    torch._dynamo.mark_dynamic(t, 0)
                    torch._dynamo.mark_dynamic(t, 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not HALF):
                fwd(*args)

        ms = timeit(run)
        print(
            f"N={n:4d} (Npad={args[0].shape[1]}, B={b:3d}): forward {ms:7.2f} ms"
            f"  = {1e3*ms/b:6.1f} us/state  -> {b/ms*1e3:7.0f} states/s"
        )
        if n == 384:
            if not compiled and not HALF:
                conv_microbench(model, args)
            profile_forward(fwd, args, f"N={n}")

    print("done")


if __name__ == "__main__":
    main()
