"""Variant 1: BF16 autocast. Easiest -- a dtype change on the autocast context.

Compares against the FP32 reference saved by 01_baseline.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bc

DEVICE = "cuda"
REF_NPZ = Path(__file__).resolve().parent / "_ref_outputs.npz"
RESULT_JSON = Path(__file__).resolve().parent / "_results_bf16.json"


def make_device_batch(arr, n, dtype=torch.float32):
    t = torch.from_numpy(np.ascontiguousarray(arr[:n])).to(DEVICE, dtype=dtype)
    return t.to(memory_format=torch.channels_last)


def run_forward(model, x):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return model.forward_policy_value(x)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = bc.build_inference_model(DEVICE, channels_last=True)
    ref = np.load(REF_NPZ)
    arr = ref["inputs"]
    pol_ref = torch.from_numpy(ref["policy"])
    val_ref = torch.from_numpy(ref["value"])

    # Correctness on the same fixed slice the ref was computed on.
    xref = make_device_batch(arr, arr.shape[0])
    o = run_forward(model, xref)
    corr = bc.correctness_report(pol_ref, val_ref, o["policy"].cpu(), o["value"].cpu())
    print(f"[corr] BF16 vs FP32: {json.dumps(corr)}", flush=True)

    results = {"corr_bf16_vs_fp32": corr, "variants": []}
    for batch in (1024, 1):
        x = make_device_batch(arr, batch)
        fn = lambda: run_forward(model, x)
        clk0 = bc.gpu_clock_mhz()
        bc.warmup(fn, seconds=6.0, min_iters=40)
        clk1 = bc.gpu_clock_mhz()
        stats = bc.time_iters(fn, iters=300)
        stats.update({
            "label": f"bf16_bs{batch}", "batch": batch,
            "forwards_per_s_mean": bc.fwd_per_s(stats["mean_ms"], batch),
            "forwards_per_s_p95": bc.fwd_per_s(stats["p95_ms"], batch),
            "clk_idle_mhz": clk0, "clk_warm_mhz": clk1,
        })
        print(f"[bf16] bs={batch} mean={stats['mean_ms']:.3f}ms p50={stats['p50_ms']:.3f} "
              f"p95={stats['p95_ms']:.3f} -> {stats['forwards_per_s_mean']:.0f} fwd/s", flush=True)
        results["variants"].append(stats)

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
