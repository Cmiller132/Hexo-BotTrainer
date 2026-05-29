"""GPU micro-benchmarks for the dense_cnn Model 1 (scratch_64) epoch cycle.

Run with the GPU FREE (live training stopped). Uses the installed editable
`hexo_models` package so it benchmarks the EXACT production network, loss, and
data path. Read-only: touches no checkpoints, configs, or run artifacts beyond
reading one real shuffled training shard.

    C:\\Python314\\python.exe analysis\\gpu_microbench.py

Outputs analysis/gpu_microbench_summary.json and prints a human summary.

Benchmarks
  A. Training step (batch 256, AMP, channels_last) steady-state ms/step, with
     variants isolating the per-step .cpu().item() sync, grad-clip, the HexConv
     mask recompute, and pure-GPU-compute vs +H2D(pageable)/+H2D(pinned).
  B. Self-play inference forward (optimized eval model) us/state at batch
     {256,1024}: full forward_policy_value vs trunk-only -> isolates the
     PolicyHead cost (the "FC head is the dominant GPU op" claim).
  C. NPZ re-decompression: current per-batch re-index vs load-once (CPU only),
     re-verifying the data-loader bug on a real shard.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import torch

from hexo_models.dense_cnn.architecture import (
    Model1Network,
    optimized_model1_for_inference,
)
from hexo_models.dense_cnn.losses import model1_loss
from hexo_models.dense_cnn.trainer import _batch_from_npz
from hexo_models.dense_cnn.replay import INPUT_KEY

HORIZONS = (1, 4, 8)
CHANNELS = 64
BLOCKS = 4
BATCH = 256
DEV = torch.device("cuda")

results: dict = {"meta": {}}


def _sync() -> None:
    torch.cuda.synchronize()


def find_shard() -> str:
    base = r"E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\shuffleddata"
    gens = sorted(
        d for d in glob.glob(os.path.join(base, "*-epoch_*")) if not d.endswith(".tmp")
    )
    for g in reversed(gens):
        shards = sorted(glob.glob(os.path.join(g, "train", "*.npz")))
        if shards:
            # prefer a mid shard with a full row count
            return shards[len(shards) // 2]
    raise SystemExit("no shard found")


def gpu_clock_info() -> dict:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return {"nvidia_smi_clocks_sm_maxsm_power_temp": out}
    except Exception as exc:  # noqa: BLE001
        return {"err": str(exc)}


# ---------------------------------------------------------------------------
# C. NPZ re-decompression (CPU) -- re-verify the data-loader bug
# ---------------------------------------------------------------------------
def bench_npz(shard: str) -> dict:
    print(f"[C] NPZ shard: {shard}")
    with np.load(shard) as d:
        rows = int(d[INPUT_KEY].shape[0])
        keys = list(d.files)
    nbatches = (rows + BATCH - 1) // BATCH
    print(f"    rows={rows} keys={keys} nbatches={nbatches}")

    # [B] current pattern: re-index every key every batch (NpzFile re-decompresses)
    t0 = time.perf_counter()
    with np.load(shard) as d:
        for s in range(0, rows, BATCH):
            e = min(s + BATCH, rows)
            _ = _batch_from_npz(d, s, e, HORIZONS)
    t_reindex = time.perf_counter() - t0

    # [C] load-once: decompress each array once, then slice in RAM
    t0 = time.perf_counter()
    with np.load(shard) as d:
        arrays = {k: d[k] for k in keys}  # one decompression each

    class _Mem:
        def __init__(self, a):
            self.a = a

        def __getitem__(self, k):
            return self.a[k]

    mem = _Mem(arrays)
    for s in range(0, rows, BATCH):
        e = min(s + BATCH, rows)
        _ = _batch_from_npz(mem, s, e, HORIZONS)
    t_loadonce = time.perf_counter() - t0

    res = {
        "shard": shard,
        "rows": rows,
        "nbatches": nbatches,
        "reindex_per_batch_s": t_reindex,
        "load_once_then_slice_s": t_loadonce,
        "speedup_x": t_reindex / max(t_loadonce, 1e-9),
    }
    print(f"    reindex={t_reindex:.2f}s  load_once={t_loadonce:.2f}s  speedup={res['speedup_x']:.1f}x")
    return res


# ---------------------------------------------------------------------------
# A. Training step
# ---------------------------------------------------------------------------
def make_train_batch(shard: str) -> dict:
    with np.load(shard) as d:
        rows = int(d[INPUT_KEY].shape[0])
        s = 0
        e = min(BATCH, rows)
        cpu = _batch_from_npz(d, s, e, HORIZONS)
    return cpu


def to_dev(cpu: dict, pin: bool) -> dict:
    out = {}
    for k, v in cpu.items():
        if pin:
            v = v.pin_memory()
        if k == "input":
            out[k] = v.to(DEV, non_blocking=True, memory_format=torch.channels_last)
        else:
            out[k] = v.to(DEV, non_blocking=True)
    _sync()
    return out


def time_loop(fn, iters=100, warmup=25) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/step


def bench_training(shard: str) -> dict:
    torch.backends.cudnn.benchmark = True
    cpu_batch = make_train_batch(shard)
    model = Model1Network(
        channels=CHANNELS, blocks=BLOCKS, short_term_value_horizons=HORIZONS
    ).to(DEV, memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def loss_of(dev_batch):
        b = dict(dev_batch)
        inp = b.pop("input")
        with torch.autocast(device_type="cuda", enabled=True):
            out = model(inp)
            loss, _ = model1_loss(
                out, b, policy_weight=1.0, value_weight=1.0,
                opp_policy_weight=0.25, short_term_value_weight=0.25,
            )
        return loss

    # Variant 1: FULL step as production does it -- includes H2D from PAGEABLE,
    # grad-clip(unscale_), AND per-step .cpu().item() sync.
    def full_step_pageable():
        dev = to_dev(cpu_batch, pin=False)
        opt.zero_grad(set_to_none=True)
        loss = loss_of(dev)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return float(loss.detach().cpu().item())

    # Pre-stage a device batch once: isolates pure GPU compute (no H2D each step).
    dev_static = to_dev(cpu_batch, pin=False)

    def step_gpu_only_sync():  # grad-clip + per-step .item() sync, data already on GPU
        opt.zero_grad(set_to_none=True)
        loss = loss_of(dev_static)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return float(loss.detach().cpu().item())

    def step_gpu_only_nosync():  # no per-step .item(); accumulate on GPU
        opt.zero_grad(set_to_none=True)
        loss = loss_of(dev_static)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    def step_gpu_only_noclip_nosync():
        opt.zero_grad(set_to_none=True)
        loss = loss_of(dev_static)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    def fwd_only():
        with torch.no_grad():
            b = dict(dev_static)
            inp = b.pop("input")
            with torch.autocast(device_type="cuda", enabled=True):
                _ = model(inp)

    out = {}
    out["full_step_pageable_h2d_ms"] = time_loop(full_step_pageable)
    out["gpu_step_with_sync_ms"] = time_loop(step_gpu_only_sync)
    out["gpu_step_no_sync_ms"] = time_loop(step_gpu_only_nosync)
    out["gpu_step_noclip_nosync_ms"] = time_loop(step_gpu_only_noclip_nosync)
    out["fwd_only_ms"] = time_loop(fwd_only)

    # H2D cost alone (pageable vs pinned), batch 256
    def h2d_pageable():
        to_dev(cpu_batch, pin=False)

    def h2d_pinned():
        to_dev(cpu_batch, pin=True)

    out["h2d_pageable_ms"] = time_loop(h2d_pageable, iters=100, warmup=20)
    out["h2d_pinned_ms"] = time_loop(h2d_pinned, iters=100, warmup=20)

    steps_per_epoch = 100000 // BATCH
    out["steps_per_epoch"] = steps_per_epoch
    out["projected_epoch_train_s_full_pageable"] = out["full_step_pageable_h2d_ms"] * steps_per_epoch / 1000.0
    out["projected_epoch_train_s_gpu_only_nosync"] = out["gpu_step_no_sync_ms"] * steps_per_epoch / 1000.0
    print("[A] training step ms:", json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# B. Inference forward (self-play evaluator)
# ---------------------------------------------------------------------------
def bench_inference() -> dict:
    torch.backends.cudnn.benchmark = True
    train_model = Model1Network(
        channels=CHANNELS, blocks=BLOCKS, short_term_value_horizons=HORIZONS
    )
    opt_model = optimized_model1_for_inference(train_model).to(DEV).eval()

    out = {}
    for bs in (256, 1024):
        x = torch.randn(bs, 13, 41, 41, device=DEV).to(memory_format=torch.channels_last)

        def full_fwd():
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=True):
                    _ = opt_model.forward_policy_value(x)

        def trunk_only():
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=True):
                    _ = opt_model.trunk(x)

        def trunk_plus_policy():
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=True):
                    f = opt_model.trunk(x)
                    _ = opt_model.policy_head(f)

        full_ms = time_loop(full_fwd, iters=100, warmup=30)
        trunk_ms = time_loop(trunk_only, iters=100, warmup=30)
        tp_ms = time_loop(trunk_plus_policy, iters=100, warmup=30)
        out[f"bs{bs}"] = {
            "full_policy_value_ms": full_ms,
            "trunk_only_ms": trunk_ms,
            "trunk_plus_policy_ms": tp_ms,
            "policy_head_ms_est": tp_ms - trunk_ms,
            "value_head_plus_overhead_ms_est": full_ms - tp_ms,
            "full_us_per_state": full_ms * 1000.0 / bs,
            "trunk_us_per_state": trunk_ms * 1000.0 / bs,
        }
    print("[B] inference ms:", json.dumps(out, indent=2))
    return out


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))
    results["meta"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "channels": CHANNELS,
        "blocks": BLOCKS,
        "batch": BATCH,
        "horizons": list(HORIZONS),
        "clocks_before": gpu_clock_info(),
    }
    shard = find_shard()
    results["C_npz"] = bench_npz(shard)
    results["A_training"] = bench_training(shard)
    results["B_inference"] = bench_inference()
    results["meta"]["clocks_after"] = gpu_clock_info()

    outp = os.path.join(os.path.dirname(__file__), "gpu_microbench_summary.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("wrote", outp)


if __name__ == "__main__":
    main()
