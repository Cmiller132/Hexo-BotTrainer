"""Evaluator-path micro-benchmarks for dense_cnn Model 1 self-play.

Pins down the MCTS GPU-evaluator binding constraint by measuring, on the free
GPU, the EXACT production evaluator (`DenseCNNInference`, optimize_for_inference,
AMP, cudnn.benchmark) used by Rust MCTS:

  1. Per-state forward latency vs batch size  -> batching efficiency curve.
  2. cudnn.benchmark autotune penalty on never-seen batch shapes (cold-epoch
     hypothesis) vs cudnn.benchmark=False.
  3. Full `evaluate_model1_payload` segment split at batch {16, 209(mean), 977(max)}:
     input-view | H2D+forward | value-decode+D2H | legal-prior gather+softmax+D2H.

Writes analysis/evaluator_microbench_summary.json. Read-only w.r.t. run state.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.inference import DenseCNNInference

DEV = torch.device("cuda")
BOARD_AREA = 41 * 41
results: dict = {"meta": {}}


def sync():
    torch.cuda.synchronize()


def build_infer() -> DenseCNNInference:
    net = Model1Network(channels=64, blocks=4, short_term_value_horizons=(1, 4, 8))
    return DenseCNNInference(net, device="cuda", amp=True, optimize_for_inference=True,
                             max_batch_size=1024)


def make_payload(b: int, legal_per_row: int = 700) -> dict:
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((b, 13, 41, 41), dtype=np.float32)
    counts = np.full((b,), legal_per_row, dtype=np.int64)
    counts = np.minimum(counts, BOARD_AREA)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    flat = rng.integers(0, BOARD_AREA, size=total, dtype=np.int64)
    return {
        "shape": (b, 13, 41, 41),
        "inputs": inputs.tobytes(),
        "legal_row_offsets": [int(x) for x in offsets.tolist()],
        "legal_flat_indices_bytes": flat.tobytes(),
    }


def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


# 1. batch-size sweep -----------------------------------------------------
def bench_batch_sweep(inf: DenseCNNInference) -> dict:
    out = {}
    for b in [1, 4, 16, 32, 64, 128, 209, 256, 512, 768, 977, 1024]:
        x = torch.randn(b, 13, 41, 41, device=DEV).to(memory_format=torch.channels_last)
        ms = time_call(lambda: inf._forward_device_inputs(x), iters=80, warmup=40)
        out[str(b)] = {"ms": ms, "us_per_state": ms * 1000.0 / b}
    print("[1] batch sweep:", json.dumps(out, indent=1))
    return out


# 2. cudnn autotune penalty ----------------------------------------------
def bench_autotune(inf: DenseCNNInference) -> dict:
    out = {}
    # Never-warmed odd batch sizes (the warmup only primed 1024).
    fresh = [173, 311, 537, 701, 823, 941]
    for mode in ("benchmark_true", "benchmark_false"):
        torch.backends.cudnn.benchmark = (mode == "benchmark_true")
        firsts, steadies = [], []
        for b in fresh:
            x = torch.randn(b, 13, 41, 41, device=DEV).to(memory_format=torch.channels_last)
            sync()
            t0 = time.perf_counter()
            inf._forward_device_inputs(x)
            sync()
            first = (time.perf_counter() - t0) * 1000.0
            # steady: next calls (same shape now cached)
            steady = time_call(lambda: inf._forward_device_inputs(x), iters=50, warmup=5)
            firsts.append(first)
            steadies.append(steady)
        out[mode] = {
            "fresh_batches": fresh,
            "first_call_ms": [round(v, 2) for v in firsts],
            "steady_ms": [round(v, 3) for v in steadies],
            "mean_first_ms": round(sum(firsts) / len(firsts), 2),
            "mean_steady_ms": round(sum(steadies) / len(steadies), 3),
            "mean_autotune_penalty_ms": round(sum(firsts) / len(firsts) - sum(steadies) / len(steadies), 2),
        }
    torch.backends.cudnn.benchmark = True
    print("[2] autotune:", json.dumps(out, indent=1))
    return out


# 3. evaluate_model1_payload segment split --------------------------------
def bench_payload_segments(inf: DenseCNNInference) -> dict:
    from hexo_models.dense_cnn.losses import decode_binned_value
    out = {}
    for b in (16, 209, 977):
        payload = make_payload(b)
        # warm full path
        e2e_ms = time_call(lambda: inf.evaluate_model1_payload(payload), iters=60, warmup=30)

        # instrumented replica mirroring inference.py:186-235
        shape = tuple(int(x) for x in payload["shape"])
        n = shape[0]
        offsets = [int(x) for x in payload["legal_row_offsets"]]
        flat_np = np.frombuffer(payload["legal_flat_indices_bytes"], dtype=np.int64)

        seg = {"view": [], "h2d_fwd": [], "valdecode_d2h": [], "gather_softmax_d2h": []}
        iters = 60
        for _ in range(iters):
            sync(); t = time.perf_counter()
            inputs = torch.frombuffer(payload["inputs"], dtype=torch.float32).reshape(shape)
            seg["view"].append(time.perf_counter() - t)

            sync(); t = time.perf_counter()
            inp_dev = inputs.to(DEV, non_blocking=True, memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
                o = inf.model.forward_policy_value(inp_dev)
            policy_batch = o["policy"].detach().float()
            value_batch = o["value"].detach().float()
            sync(); seg["h2d_fwd"].append(time.perf_counter() - t)

            t = time.perf_counter()
            _ = decode_binned_value(value_batch).cpu().contiguous()
            sync(); seg["valdecode_d2h"].append(time.perf_counter() - t)

            t = time.perf_counter()
            counts = torch.as_tensor([offsets[i + 1] - offsets[i] for i in range(n)], dtype=torch.long)
            row_ids = torch.repeat_interleave(torch.arange(n, dtype=torch.long), counts)
            flat_indices = torch.from_numpy(flat_np).to(policy_batch.device, non_blocking=True)
            row_ids_d = row_ids.to(policy_batch.device, non_blocking=True)
            selected = policy_batch[row_ids_d, flat_indices]
            maxr = torch.full((n,), float("-inf"), dtype=selected.dtype, device=selected.device)
            maxr.scatter_reduce_(0, row_ids_d, selected, reduce="amax", include_self=True)
            ex = torch.exp(selected - maxr[row_ids_d])
            sm = torch.zeros((n,), dtype=selected.dtype, device=selected.device)
            sm.scatter_add_(0, row_ids_d, ex)
            _ = (ex / sm[row_ids_d]).cpu().contiguous()
            sync(); seg["gather_softmax_d2h"].append(time.perf_counter() - t)

        def ms(key):
            return sum(seg[key]) / len(seg[key]) * 1000.0
        out[str(b)] = {
            "e2e_ms": e2e_ms,
            "view_ms": ms("view"),
            "h2d_fwd_ms": ms("h2d_fwd"),
            "valdecode_d2h_ms": ms("valdecode_d2h"),
            "gather_softmax_d2h_ms": ms("gather_softmax_d2h"),
            "sum_segments_ms": ms("view") + ms("h2d_fwd") + ms("valdecode_d2h") + ms("gather_softmax_d2h"),
            "e2e_us_per_state": e2e_ms * 1000.0 / b,
        }
    print("[3] payload segments:", json.dumps(out, indent=1))
    return out


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    inf = build_infer()
    results["meta"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    results["batch_sweep"] = bench_batch_sweep(inf)
    results["payload_segments"] = bench_payload_segments(inf)
    results["autotune"] = bench_autotune(inf)
    outp = os.path.join(os.path.dirname(__file__), "evaluator_microbench_summary.json")
    json.dump(results, open(outp, "w", encoding="utf-8"), indent=2)
    print("wrote", outp)


if __name__ == "__main__":
    main()
