"""Attribution: is the GPU forward really the bottleneck behind ~1287 fwd/s?

Decompose the production Python/Torch evaluator boundary at bs=1024:
  (A) raw forward_policy_value only (compute on a device-resident tensor),
  (B) full DenseCNNInference.evaluate_model1_payload (the real MCTS callback:
      frombuffer + H2D channels_last + bucket pad + forward + value decode +
      legal-flat gather/softmax + D2H + tobytes).

If (B) >> (A), the forward is NOT the sole cost at the boundary and a faster
forward yields sub-linear end-to-end gains. Also runs the real self-play probe
(performance._benchmark_selfplay-style) to anchor end-to-end pos/s.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bc

DEVICE = "cuda"
RESULT_JSON = Path(__file__).resolve().parent / "_results_attribution.json"


def build_payload(arr, n):
    """Construct a real evaluator payload from representative states' legal flats."""
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn import rust_bridge
    import random

    # Re-derive legal flats for n states by replaying random games (same as the
    # input pool generation), so the gather path sees realistic legal-set sizes.
    rng = random.Random(999)
    states = []
    gi = 0
    while len(states) < n:
        st = engine.new_game(seed=120_000 + gi); gi += 1
        for _ in range(rng.randint(2, 120)):
            if engine.terminal(st) is not None:
                break
            aids = engine.legal_action_ids(st)
            if not aids:
                break
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(aids))))
        if engine.terminal(st) is None:
            states.append(engine.clone_state(st))
    payload = rust_bridge.model1_batch_inputs(states[:n])
    shape = tuple(int(x) for x in payload["shape"])
    inputs_bytes = bytes(payload["inputs"])
    flats_rows = [list(int(i) for i in row) for row in payload["legal_flat_indices"]]
    offsets = [0]
    flat_concat = []
    for row in flats_rows:
        flat_concat.extend(row)
        offsets.append(len(flat_concat))
    flat_arr = np.array(flat_concat, dtype=np.int64)
    return {
        "shape": shape,
        "inputs": inputs_bytes,
        "legal_flat_indices_bytes": flat_arr.tobytes(),
        "legal_row_offsets": offsets,
    }, shape


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn.inference import DenseCNNInference

    base = bc.build_training_model()  # DenseCNNInference will build the optimized clone itself
    inf = DenseCNNInference(base, device=DEVICE, amp=True, max_batch_size=1024, optimize_for_inference=True)

    arr = bc.generate_inputs(2048)
    N = 1024
    payload, shape = build_payload(arr, N)
    print(f"[payload] shape={shape} flats={len(payload['legal_flat_indices_bytes'])//8} "
          f"avg_legal={len(payload['legal_flat_indices_bytes'])//8/N:.1f}", flush=True)

    results = {}

    # (A) raw forward only (device-resident tensor, fp16 autocast == production)
    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).to(DEVICE).to(memory_format=torch.channels_last)
    def raw_fwd():
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                model = inf.model
                return model.forward_policy_value(x)
    bc.warmup(raw_fwd, seconds=5.0, min_iters=40)
    a = bc.time_iters(raw_fwd, iters=300)
    a["forwards_per_s_mean"] = bc.fwd_per_s(a["mean_ms"], N)
    results["raw_forward_only"] = a
    print(f"[A raw fwd] mean={a['mean_ms']:.3f}ms -> {a['forwards_per_s_mean']:.0f} fwd/s", flush=True)

    # (B) full evaluator callback (the real MCTS boundary). Timed on the CPU
    # wall clock because it includes D2H/H2D + python; sync inside the call.
    def full_cb():
        return inf.evaluate_model1_payload(payload)
    # warmup
    for _ in range(40):
        full_cb()
    bc.cuda_sync()
    lat = []
    for _ in range(300):
        t0 = time.perf_counter()
        full_cb()
        bc.cuda_sync()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat); lat.sort()
    b = {"mean_ms": float(lat.mean()), "std_ms": float(lat.std()),
         "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
         "min_ms": float(lat.min()), "iters": 300}
    b["forwards_per_s_mean"] = bc.fwd_per_s(b["mean_ms"], N)
    results["full_evaluator_callback"] = b
    print(f"[B full cb] mean={b['mean_ms']:.3f}ms -> {b['forwards_per_s_mean']:.0f} fwd/s", flush=True)

    results["forward_fraction_of_callback"] = a["mean_ms"] / b["mean_ms"]
    print(f"[attribution] forward is {results['forward_fraction_of_callback']*100:.1f}% of the "
          f"evaluator callback at bs={N}", flush=True)

    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
