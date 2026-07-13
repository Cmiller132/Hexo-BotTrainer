"""hexfield_eq training-step (fwd+bwd) kernel profile at live main_3 shapes.

Replicates the production trainer's inner loop: autocast fp16 forward of the
full head set + scaled backward at PAIR_BUDGET=1.6e7 microbuckets (B=59 @
Npad=512), A5+lut2 arch (c=192, CCACCACA, RAYTAP=both, RAYTAP_LUT=additive).

Usage (WSL, GPU free):
  PYTHONPATH=packages/hexfield_eq/python /root/.venvs/hexgt-build/bin/python \
      scripts/_eq_train_step_profile.py [compiled] [profile]

`compiled` wraps forward in torch.compile like the trainer (B dynamic-ish,
Npad static); `profile` emits a torch.profiler table of the top CUDA kernels.
"""

from __future__ import annotations

import os
import sys
import time

# Live main_3 architecture (scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env)
_ENV = {
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_GROUP_ORDER": "12",
    "HEXFIELD_EQ_C_ORBIT": "16",
    "HEXFIELD_EQ_ATTENTION_HEADS": "3",
    "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
    "HEXFIELD_EQ_TRUNK": "CCACCACA",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_REG_TOK_READ": "0",
    "HEXFIELD_EQ_FEATURE_VERSION": "2",
    "HEXFIELD_EQ_RAYTAP": "both",
    "HEXFIELD_EQ_RAYTAP_LUT": "additive",
    "HEXFIELD_TRAIN_FLEX": "1",
}
for k, v in _ENV.items():
    os.environ.setdefault(k, v)

import numpy as np
import torch
import torch._dynamo

from hexfield_eq.constants import DIRECTIONS, NUM_FEATURES, RAYLEN_SLOTS
from hexfield_eq.model import HexfieldNet

PAIR_BUDGET = 1.6e7
QUANT = 64
DISK_R = 4


def hexdist(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


DISK = [
    (dq, dr)
    for dq in range(-DISK_R, DISK_R + 1)
    for dr in range(-DISK_R, DISK_R + 1)
    if hexdist(dq, dr) <= DISK_R
]


def make_blob(n_target: int, rng: np.random.Generator):
    stones = [(0, 0)]
    support: set = set(DISK)
    while len(support) < n_target:
        base = stones[rng.integers(len(stones))]
        d = DIRECTIONS[rng.integers(6)]
        s = (base[0] + d[0], base[1] + d[1])
        if s in stones:
            continue
        stones.append(s)
        for dq, dr in DISK:
            support.add((s[0] + dq, s[1] + dr))
    cells = sorted(support)[:n_target]
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
    feats = torch.zeros(b, npad, NUM_FEATURES, dtype=torch.float32)
    nbrs = torch.full((b, npad, 6), npad, dtype=torch.long)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    coords = torch.zeros(b, npad, 2, dtype=torch.long)
    raylen = torch.zeros(b, npad, RAYLEN_SLOTS, dtype=torch.uint8)
    for k in range(4):
        c, nb = make_blob(n, rng)
        rl = rng.integers(0, 6, size=(n, RAYLEN_SLOTS), dtype=np.uint8)
        for i in range(k, b, 4):
            feats[i, :n] = torch.from_numpy(
                rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
            )
            nbrs[i, :n] = torch.from_numpy(nb)
            nbrs[i, :n][nbrs[i, :n] == n] = npad
            mask[i, :n] = True
            coords[i, :n] = torch.from_numpy(c)
            raylen[i, :n] = torch.from_numpy(rl)
    return (
        feats.to(device),
        nbrs.to(device),
        mask.to(device),
        coords.to(device),
        raylen.to(device),
    )


MARK = False


def step(model, scaler, args):
    feats, nbrs, mask, coords, raylen = args
    if MARK:
        for t in (feats, nbrs, mask, coords):
            torch._dynamo.maybe_mark_dynamic(t, 0)
            torch._dynamo.mark_static(t, 1)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(feats, nbrs, mask, coords, raylen=raylen)
    loss = sum(o.float().mean() for o in out.values())
    scaler.scale(loss).backward()
    return loss


def main() -> None:
    global MARK
    compiled = "compiled" in sys.argv[1:]
    profile = "profile" in sys.argv[1:]
    device = torch.device("cuda")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = HexfieldNet().to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"torch {torch.__version__}  params {n_params}  compiled {compiled}")

    if compiled:
        torch._dynamo.config.cache_size_limit = 64
        model_fwd = torch.compile(model.forward)
        MARK = True
    else:
        model_fwd = model.forward

    class _Wrap:
        def __call__(self, *a, **kw):
            return model_fwd(*a, **kw)

    wrapped = _Wrap()

    # Production-like microbucket: Npad=512, B = 1.6e7 / (512+8)^2 = 59.
    s = 512 + 8
    b512 = max(2, int(PAIR_BUDGET / (s * s)))
    args = make_batch(b512, 470, device, rng)
    scaler = torch.amp.GradScaler("cuda")
    params = list(model.parameters())

    t0 = time.perf_counter()
    for _ in range(3):
        step(wrapped, scaler, args)
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    print(f"warmup done in {time.perf_counter()-t0:.0f}s")

    iters = 8
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step(wrapped, scaler, args)
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    rows = args[0].shape[0]
    print(
        f"microbucket Npad=512 B={rows}: {ms:7.1f} ms fwd+bwd "
        f"({1e3*ms/rows:.0f} us/row)  peak_mem "
        f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
    )

    if profile:
        from torch.profiler import ProfilerActivity, profile as tprofile

        with tprofile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            for _ in range(3):
                step(wrapped, scaler, args)
                for p in params:
                    p.grad = None
            torch.cuda.synchronize()
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40,
                max_name_column_width=80,
            )
        )
    print("done")


if __name__ == "__main__":
    main()
