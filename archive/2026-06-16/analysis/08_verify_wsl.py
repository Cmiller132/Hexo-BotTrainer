"""Second-way verification of the two headline winners (compile, TRT FP16).

Fresh process. For each of {eager fp16, torch.compile fp16, TRT fp16} at the
production buckets 128/256, time the SAME forward TWO independent ways:
  (1) CUDA events (as in the main runs), and
  (2) plain wall-clock: sync -> loop N -> sync -> total/N.
If the two methods agree and reproduce the original numbers, the headline
speedups are real (not an event-overlap or single-run artifact).
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
HERE = Path(__file__).resolve().parent
REF_NPZ = HERE / "_ref_outputs.npz"
RESULT_JSON = HERE / "_results_verify.json"
BATCHES = [128, 256]


class PVWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_policy_value(x)


def wallclock(fn, iters=300):
    bc.cuda_sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    bc.cuda_sync()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0  # ms


def both_ways(fn, bs, label, out):
    bc.warmup(fn, seconds=5.0, min_iters=40)
    ev = bc.time_iters(fn, iters=300)["mean_ms"]
    wc = wallclock(fn, iters=300)
    rec = {"label": label, "batch": bs, "event_ms": ev, "wallclock_ms": wc,
           "event_fwd_s": bc.fwd_per_s(ev, bs), "wallclock_fwd_s": bc.fwd_per_s(wc, bs),
           "method_agreement_pct": 100.0 * abs(ev - wc) / ((ev + wc) / 2)}
    out.append(rec)
    print(f"[{label}] bs={bs} event={ev:.4f}ms ({rec['event_fwd_s']:.0f}/s) "
          f"wallclock={wc:.4f}ms ({rec['wallclock_fwd_s']:.0f}/s) "
          f"agree={rec['method_agreement_pct']:.1f}%", flush=True)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[verify] torch={torch.__version__}", flush=True)

    model = bc.build_inference_model(DEVICE, channels_last=True)
    wrap = PVWrap(model).to(DEVICE).eval()
    arr = np.load(REF_NPZ)["inputs"]

    def mk(bs):
        idx = np.arange(bs) % arr.shape[0]
        return torch.from_numpy(np.ascontiguousarray(arr[idx])).to(DEVICE).to(memory_format=torch.channels_last)

    out = []

    # Eager fp16.
    for bs in BATCHES:
        x = mk(bs)
        fn = lambda: (torch.autocast("cuda", dtype=torch.float16).__enter__(), wrap(x))[-1]
        def eager():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return wrap(x)
        both_ways(eager, bs, f"eager_fp16_bs{bs}", out)

    # torch.compile fp16.
    compiled = torch.compile(wrap, mode="max-autotune", dynamic=False)
    for bs in BATCHES:
        x = mk(bs)
        def comp():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return compiled(x)
        both_ways(comp, bs, f"compile_fp16_bs{bs}", out)

    # TRT fp16 (rebuild engine from cached onnx via 07's helpers).
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("trt07", str(HERE / "07_trt.py"))
        trt07 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trt07)
        model_plain = bc.build_inference_model(DEVICE, channels_last=False)
        if not trt07.onnx_path("fp16").exists():
            trt07.export_onnx(model_plain, "fp16")
        engine, _ = trt07.build_engine("fp16")
        run = trt07.trt_runner(engine)
        for bs in BATCHES:
            idx = np.arange(bs) % arr.shape[0]
            xt = torch.from_numpy(np.ascontiguousarray(arr[idx])).to(DEVICE)
            both_ways(lambda: run(xt), bs, f"trt_fp16_bs{bs}", out)
    except Exception as e:
        print(f"[verify] TRT leg skipped: {e}", flush=True)

    RESULT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
