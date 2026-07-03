"""Training-step (fwd+bwd) microbench at production shapes (c=128, radius-4).

Replicates the production trainer's inner loop shape: autocast fp16 forward of
the FULL head set + scaled backward, grad-accumulated over microbuckets, at
PAIR_BUDGET-sized buckets (B~74 @ Npad=512). Compares:
  eager + train_flex            (production today)
  eager + train_flex_pair       (precomputed-pair score_mod, grad table2)
  eager materialized            (_BiasGather reference)
  compiled + train_flex[_pair]  (torch.compile(dynamic=True) over forward)

Usage (WSL, GPU free):
  HEXFIELD_CHANNELS=128 [HEXFIELD_TRAIN_FLEX=1] [HEXFIELD_TRAIN_FLEX_PAIR=1] \
  PYTHONPATH=... python scripts/_hexfield_train_step_bench.py [compiled] [shapes]

`shapes` adds a multi-shape pass (varied B/Npad like real microbuckets) to
expose compile/flex recompile behavior.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HEXFIELD_CHANNELS", "128")

import numpy as np
import torch
import torch._dynamo

from hexfield.model import HexfieldNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hexfield_main6_profile import make_batch  # noqa: E402

PAIR_BUDGET = 2.0e7
PAD_QUANTUM = 256


def bucket_b(npad: int) -> int:
    s = npad + 8
    return max(2, int(PAIR_BUDGET / (s * s)))


MARK = False  # set in main: trainer-style B-dynamic / Npad-static marking


def step(model, scaler, args, loss_weights):
    feats, nbrs, mask, coords = args
    if MARK:
        for t in args:
            torch._dynamo.maybe_mark_dynamic(t, 0)
            torch._dynamo.mark_static(t, 1)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(feats, nbrs, mask, coords)
    # Loss proxy: mean over every head output (exercises all backward paths,
    # incl. the bias-table gradient through the attention score).
    loss = sum(w * o.float().mean() for (w, o) in zip(loss_weights, out.values()))
    scaler.scale(loss).backward()
    return loss


def timeit_steps(model, args_list, warmup=3, iters=12):
    scaler = torch.amp.GradScaler("cuda")
    params = list(model.parameters())
    lw = [1.0] * 16
    for _ in range(warmup):
        for a in args_list:
            step(model, scaler, a, lw)
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for a in args_list:
            step(model, scaler, a, lw)
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    global MARK
    compiled = "compiled" in sys.argv[1:]
    shapes = "shapes" in sys.argv[1:]
    big = "big" in sys.argv[1:]
    device = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = HexfieldNet().to(device).train()
    print(
        f"torch {torch.__version__}  c {model.stem.out_channels}  "
        f"train_flex {model._train_flex}  train_flex_pair {model._train_flex_pair}  "
        f"compiled {compiled}  big {big}"
    )
    if compiled:
        torch._dynamo.config.cache_size_limit = 64
        # Trainer-style: B symbolic, Npad static per PAD_QUANTUM multiple.
        model.forward = torch.compile(model.forward)
        MARK = True

    budget = 8.0e7 if big else PAIR_BUDGET
    # Single production-like bucket: Npad=512, B under the budget.
    s = 512 + 8
    b512 = max(2, int(budget / (s * s)))
    args = make_batch(b512, 470, device, rng)  # 470 -> Npad 512 under the 64-quantum
    ms = timeit_steps(model, [args])
    rows = args[0].shape[0]
    print(f"single bucket Npad=512 B={rows}: {ms:7.1f} ms/microbucket  ({1e3*ms/rows:.0f} us/row)")

    if shapes:
        # Varied microbuckets: the shape stream the real trainer produces.
        buckets = []
        for npad_n, n in ((256, 210), (512, 470), (768, 700), (512, 500), (256, 240)):
            bb = bucket_b(-(-n // PAD_QUANTUM) * PAD_QUANTUM)
            bb = max(2, bb - int(rng.integers(0, max(2, bb // 3))))  # vary B like real buckets
            buckets.append(make_batch(bb, n, device, rng))
        t0 = time.perf_counter()
        ms = timeit_steps(model, buckets, warmup=2, iters=6)
        total_rows = sum(a[0].shape[0] for a in buckets)
        print(
            f"5 varied buckets ({total_rows} rows): {ms:7.1f} ms/pass  "
            f"({1e3*ms/total_rows:.0f} us/row)  [wall incl. warmup {time.perf_counter()-t0:.0f}s]"
        )
    print("done")


if __name__ == "__main__":
    main()
