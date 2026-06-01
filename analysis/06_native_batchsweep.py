"""Authoritative native-Windows sweep: fp32 / fp16 / bf16 across the REAL
production batch shapes.

05_selfplay_attribution showed the real MCTS leaf batch is mean~99 (p50 70,
p95 228, max 245) -> power-of-2 bucketed to 128 / 256. bs1024 almost never
occurs. So the production-relevant forward shapes are 128 and 256; we also keep
1 (single-game eval) and 1024 (max bucket / cross-check vs 01_baseline).
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
RESULT_JSON = Path(__file__).resolve().parent / "_results_native_sweep.json"
BATCHES = [1, 128, 256, 1024]
DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def make_batch(arr, n):
    # Real inputs cycle through the pool so a batch is representative, not one
    # repeated row (guards against any input-reuse caching artifact).
    idx = np.arange(n) % arr.shape[0]
    t = torch.from_numpy(np.ascontiguousarray(arr[idx])).to(DEVICE, dtype=torch.float32)
    return t.to(memory_format=torch.channels_last)


def run(model, x, amp_dtype):
    with torch.inference_mode():
        if amp_dtype is None:
            return model.forward_policy_value(x)
        with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
            return model.forward_policy_value(x)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}", flush=True)

    model = bc.build_inference_model(DEVICE, channels_last=True)
    ref = np.load(REF_NPZ)
    arr = ref["inputs"]
    pol_ref = torch.from_numpy(ref["policy"])
    val_ref = torch.from_numpy(ref["value"])
    n_corr = arr.shape[0]

    results = {"correctness": {}, "variants": []}

    # Correctness per dtype on the fixed slice.
    xc = make_batch(arr, n_corr)
    for name, dt in DTYPES.items():
        o = run(model, xc, dt)
        results["correctness"][name] = bc.correctness_report(pol_ref, val_ref, o["policy"].cpu(), o["value"].cpu())
        print(f"[corr {name}] {json.dumps(results['correctness'][name])}", flush=True)

    for bs in BATCHES:
        x = make_batch(arr, bs)
        for name, dt in DTYPES.items():
            fn = lambda: run(model, x, dt)
            clk0 = bc.gpu_clock_mhz()
            bc.warmup(fn, seconds=5.0, min_iters=40)
            clk1 = bc.gpu_clock_mhz()
            s = bc.time_iters(fn, iters=300)
            s.update({"dtype": name, "batch": bs,
                      "forwards_per_s_mean": bc.fwd_per_s(s["mean_ms"], bs),
                      "forwards_per_s_p95": bc.fwd_per_s(s["p95_ms"], bs),
                      "clk_idle_mhz": clk0, "clk_warm_mhz": clk1})
            results["variants"].append(s)
            print(f"[{name} bs{bs}] mean={s['mean_ms']:.4f}ms p50={s['p50_ms']:.4f} p95={s['p95_ms']:.4f} "
                  f"std={s['std_ms']:.4f} -> {s['forwards_per_s_mean']:.0f} fwd/s (clk {clk0}->{clk1})", flush=True)

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
