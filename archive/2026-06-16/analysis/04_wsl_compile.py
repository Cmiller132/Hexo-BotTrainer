"""Variant 2: torch.compile, run in WSL2 (per the user). torch 2.11+cu128.

Measures, all on the same GPU:
  * WSL eager FP16 baseline (to separate the WSL/torch-version effect from the
    compile gain -- the native baseline is torch 2.10+cu126).
  * torch.compile FP16 (static shapes per bucket: 1024 and 1).
Correctness gate against a WSL-LOCAL FP32 reference (apples-to-apples on this
runtime), plus a cross-check that WSL-FP32 == native-FP32 (saved by 01).

Run: wsl -d Ubuntu-24.04 -- /root/.venvs/hexo-bottrainer-wsl/bin/python \
       /mnt/e/Hexo-BotTrainer/analysis/04_wsl_compile.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bc

DEVICE = "cuda"
REF_NPZ = Path(__file__).resolve().parent / "_ref_outputs.npz"
RESULT_JSON = Path(__file__).resolve().parent / "_results_wsl_compile.json"


class PVWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_policy_value(x)


def make_batch(arr, n, dtype=torch.float32):
    t = torch.from_numpy(np.ascontiguousarray(arr[:n])).to(DEVICE, dtype=dtype)
    return t.to(memory_format=torch.channels_last)


def run(model, x, amp_dtype):
    with torch.inference_mode():
        if amp_dtype is None:
            return model(x)
        with torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
            return model(x)


def bench(fn, batch, label, warm_s=6.0):
    clk0 = bc.gpu_clock_mhz()
    bc.warmup(fn, seconds=warm_s, min_iters=40)
    clk1 = bc.gpu_clock_mhz()
    s = bc.time_iters(fn, iters=300)
    s.update({"label": label, "batch": batch,
              "forwards_per_s_mean": bc.fwd_per_s(s["mean_ms"], batch),
              "forwards_per_s_p95": bc.fwd_per_s(s["p95_ms"], batch),
              "clk_idle_mhz": clk0, "clk_warm_mhz": clk1})
    print(f"[{label}] bs={batch} mean={s['mean_ms']:.3f}ms p50={s['p50_ms']:.3f} "
          f"p95={s['p95_ms']:.3f} -> {s['forwards_per_s_mean']:.0f} fwd/s", flush=True)
    return s


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[wsl] torch={torch.__version__} cuda={torch.version.cuda} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)

    model = bc.build_inference_model(DEVICE, channels_last=True)
    wrap = PVWrap(model).to(DEVICE).eval()

    ref = np.load(REF_NPZ)
    arr = ref["inputs"]
    nat_pol = torch.from_numpy(ref["policy"])
    nat_val = torch.from_numpy(ref["value"])
    n_corr = arr.shape[0]

    # WSL-local FP32 reference + cross-check vs native FP32.
    xref = make_batch(arr, n_corr, torch.float32)
    with torch.inference_mode():
        w32 = wrap(xref)
    wsl_pol = w32["policy"].float().cpu()
    wsl_val = w32["value"].float().cpu()
    cross = {
        "wsl_fp32_vs_native_fp32_policy_max_abs": bc.max_abs_err(nat_pol, wsl_pol),
        "wsl_fp32_vs_native_fp32_value_max_abs": bc.max_abs_err(nat_val, wsl_val),
    }
    print(f"[cross] {json.dumps(cross)}", flush=True)

    results = {"torch": torch.__version__, "cross_check": cross, "variants": [], "correctness": {}}

    BATCHES = [1, 128, 256, 1024]  # 128/256 are the real production buckets (see 05)

    # WSL eager FP16 baseline (isolates the WSL/torch-2.11 effect from compile).
    for batch in BATCHES:
        x = make_batch(arr, batch, torch.float32)
        results["variants"].append(bench(lambda: run(wrap, x, torch.float16), batch, f"wsl_eager_fp16_bs{batch}"))

    # torch.compile FP16. Compile per static bucket shape; time compile wall.
    compiled = torch.compile(wrap, mode="max-autotune", dynamic=False)
    compile_times = {}
    for batch in BATCHES:
        x = make_batch(arr, batch, torch.float32)
        t0 = time.perf_counter()
        run(compiled, x, torch.float16)  # triggers compile for this shape
        bc.cuda_sync()
        compile_times[f"bs{batch}"] = time.perf_counter() - t0
        print(f"[compile] bs={batch} first-call(compile) {compile_times[f'bs{batch}']:.1f}s", flush=True)
    results["compile_wall_seconds"] = compile_times

    # Correctness: compiled FP16 vs WSL FP32.
    xc = make_batch(arr, n_corr, torch.float32)
    oc = run(compiled, xc, torch.float16)
    corr = bc.correctness_report(wsl_pol, wsl_val, oc["policy"].cpu(), oc["value"].cpu())
    results["correctness"]["compile_fp16_vs_wsl_fp32"] = corr
    print(f"[corr] compile_fp16 vs wsl_fp32: {json.dumps(corr)}", flush=True)

    for batch in BATCHES:
        x = make_batch(arr, batch, torch.float32)
        results["variants"].append(bench(lambda: run(compiled, x, torch.float16), batch, f"wsl_compile_fp16_bs{batch}"))

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
