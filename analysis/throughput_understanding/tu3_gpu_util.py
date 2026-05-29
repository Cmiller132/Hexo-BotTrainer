"""Q3 + Q4: GPU saturation evidence (no ncu available, so multiple proxies).

(1) Fine effective-batch sweep of the production FP16 forward: per-sample
    throughput (fwd/s) vs batch from 16..1024. If fwd/s rises then plateaus, the
    plateau is the saturation knee -> tells us whether fatter forwards (Q4) can
    raise throughput or the model already saturates at the real ~99/128.
(2) torch.profiler kernel breakdown at bs128: sum of GPU-kernel time vs wall
    (in-kernel duty), kernel count per forward, and the top kernels. If wall >>
    kernel-sum the forward is launch/sync-bound (idle SMs); if kernels are many
    and tiny the model is latency-bound (which fusion/compile/TRT fix) rather
    than compute-bound (which only a bigger/faster GPU fixes).

All on the EXACT production inference model (folded, channels_last, autocast fp16).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tu3_gpu_util.json"

BATCHES = [16, 32, 64, 96, 99, 128, 160, 192, 256, 384, 512, 768, 1024]


def build_inf_model():
    import tomllib
    from hexo_models.dense_cnn.architecture import optimized_model1_for_inference
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin

    raw = tomllib.loads(CONFIG.read_text())
    section = raw["model"]["config"]
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval()
    opt = optimized_model1_for_inference(model).to("cuda", memory_format=torch.channels_last).eval()
    return opt


def make_x(bs):
    # representative-ish nonzero input; values don't affect conv timing.
    g = torch.Generator().manual_seed(7)
    x = (torch.rand((bs, 13, 41, 41), generator=g) > 0.7).float()
    return x.to("cuda").to(memory_format=torch.channels_last)


def run(model, x):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return model.forward_policy_value(x)


def warm(fn, seconds=4.0, min_iters=30):
    import time
    torch.cuda.synchronize(); t0 = time.perf_counter(); it = 0
    while True:
        fn(); it += 1
        if it >= min_iters and time.perf_counter() - t0 >= seconds:
            break
    torch.cuda.synchronize()


def time_ms(fn, iters=200):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for i in range(iters):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    lat = np.array([a.elapsed_time(b) for a, b in zip(s, e)])
    return float(lat.mean()), float(np.percentile(lat, 95))


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_inf_model()
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    out = {"sweep": [], "profile_bs128": None}

    # (1) batch sweep
    for bs in BATCHES:
        x = make_x(bs)
        fn = lambda: run(model, x)
        warm(fn, seconds=3.0, min_iters=30)
        mean_ms, p95 = time_ms(fn, iters=200)
        rec = {"batch": bs, "mean_ms": mean_ms, "p95_ms": p95,
               "fwd_per_s": bs / (mean_ms / 1000.0), "us_per_sample": mean_ms * 1000.0 / bs}
        out["sweep"].append(rec)
        print(f"  bs={bs:>4d} mean={mean_ms:8.4f}ms fwd/s={rec['fwd_per_s']:8.0f} "
              f"us/sample={rec['us_per_sample']:7.2f}", flush=True)

    # (2) profiler at bs128
    x = make_x(128)
    fn = lambda: run(model, x)
    warm(fn, seconds=3.0, min_iters=30)
    import time
    torch.cuda.synchronize()
    n_prof = 50
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False) as prof:
        t0 = time.perf_counter()
        for _ in range(n_prof):
            fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    ka = prof.key_averages()
    total_cuda_us = sum(getattr(item, "self_device_time_total", getattr(item, "self_cuda_time_total", 0)) for item in ka)
    n_kernels = sum(item.count for item in ka)
    top = sorted(ka, key=lambda i: getattr(i, "self_device_time_total", getattr(i, "self_cuda_time_total", 0)), reverse=True)[:12]
    out["profile_bs128"] = {
        "iters": n_prof, "wall_s": wall, "wall_ms_per_fwd": wall / n_prof * 1000.0,
        "total_cuda_us": total_cuda_us, "cuda_ms_per_fwd": total_cuda_us / 1000.0 / n_prof,
        "in_kernel_duty": (total_cuda_us / 1000.0) / (wall * 1000.0),
        "kernels_per_fwd": n_kernels / n_prof,
        "top_kernels": [
            {"name": i.key[:60],
             "cuda_ms_per_fwd": getattr(i, "self_device_time_total", getattr(i, "self_cuda_time_total", 0)) / 1000.0 / n_prof,
             "count_per_fwd": i.count / n_prof}
            for i in top
        ],
    }
    p = out["profile_bs128"]
    print(f"\n[profile bs128] wall={p['wall_ms_per_fwd']:.3f}ms/fwd cuda={p['cuda_ms_per_fwd']:.3f}ms/fwd "
          f"in_kernel_duty={p['in_kernel_duty']*100:.1f}% kernels/fwd={p['kernels_per_fwd']:.0f}", flush=True)
    for k in p["top_kernels"]:
        print(f"    {k['cuda_ms_per_fwd']*1000:7.1f}us/fwd x{k['count_per_fwd']:.0f}  {k['name']}", flush=True)

    RESULT.write_text(json.dumps(out, indent=2))
    print(f"[tu3] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
