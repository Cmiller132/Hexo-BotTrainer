"""Baseline: FP32 reference + production FP16/AMP forward (native Windows).

Establishes the numbers every other variant is measured against:
  * FP32 reference outputs (saved to _ref_outputs.npz) for the correctness gate.
  * FP32 forward throughput/latency (correctness reference timing).
  * FP16/autocast forward throughput/latency == the PRODUCTION evaluator forward.
Both at the production inference batch (1024) and a single-game batch (1).
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
CORR_N = 1024  # rows used for the correctness gate / saved reference
RESULT_JSON = Path(__file__).resolve().parent / "_results_baseline.json"
REF_NPZ = Path(__file__).resolve().parent / "_ref_outputs.npz"


def make_device_batch(arr: np.ndarray, n: int, dtype=torch.float32) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr[:n])).to(DEVICE, dtype=dtype)
    return t.to(memory_format=torch.channels_last)


def run_forward(model, x, amp: bool):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=amp, dtype=torch.float16):
            out = model.forward_policy_value(x)
        return out


def bench_variant(model, x, amp: bool, label: str) -> dict:
    fn = lambda: run_forward(model, x, amp)
    clk_idle = bc.gpu_clock_mhz()
    bc.warmup(fn, seconds=6.0, min_iters=40)
    clk_warm = bc.gpu_clock_mhz()
    stats = bc.time_iters(fn, iters=300)
    batch = x.shape[0]
    stats.update({
        "label": label, "batch": batch, "amp": amp,
        "forwards_per_s_mean": bc.fwd_per_s(stats["mean_ms"], batch),
        "forwards_per_s_p95": bc.fwd_per_s(stats["p95_ms"], batch),
        "clk_idle_mhz": clk_idle, "clk_warm_mhz": clk_warm,
    })
    print(f"[{label}] bs={batch} amp={amp} mean={stats['mean_ms']:.3f}ms "
          f"p50={stats['p50_ms']:.3f} p95={stats['p95_ms']:.3f} std={stats['std_ms']:.3f} "
          f"-> {stats['forwards_per_s_mean']:.0f} fwd/s (clk {clk_idle}->{clk_warm}MHz)", flush=True)
    return stats


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}", flush=True)

    model = bc.build_inference_model(DEVICE, channels_last=True)
    arr = bc.generate_inputs(2048)

    # ---- FP32 reference outputs on the fixed correctness slice ----
    xref = make_device_batch(arr, CORR_N, torch.float32)
    with torch.inference_mode():
        ref = model.forward_policy_value(xref)
    pol_ref = ref["policy"].float().cpu().contiguous()
    val_ref = ref["value"].float().cpu().contiguous()
    np.savez(REF_NPZ, policy=pol_ref.numpy(), value=val_ref.numpy(), inputs=arr[:CORR_N])
    print(f"[ref] saved FP32 reference outputs policy={tuple(pol_ref.shape)} value={tuple(val_ref.shape)}", flush=True)

    # Self-consistency: FP16 vs FP32 correctness on the same slice.
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            o16 = model.forward_policy_value(xref)
    corr_fp16 = bc.correctness_report(pol_ref, val_ref, o16["policy"].cpu(), o16["value"].cpu())
    print(f"[corr] FP16 vs FP32: {json.dumps(corr_fp16)}", flush=True)

    results = {"corr_fp16_vs_fp32": corr_fp16, "variants": []}

    # ---- Timing: bs=1024 and bs=1, fp32 and fp16 ----
    for batch in (1024, 1):
        x32 = make_device_batch(arr, batch, torch.float32)
        results["variants"].append(bench_variant(model, x32, amp=False, label=f"fp32_bs{batch}"))
        results["variants"].append(bench_variant(model, x32, amp=True, label=f"fp16_amp_bs{batch}"))

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
