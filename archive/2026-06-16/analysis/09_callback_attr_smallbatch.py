"""Forward-vs-callback attribution at the REAL self-play batch sizes.

03 did this at bs1024 (forward = 92% of callback). The real MCTS leaf batches
are ~70-99 (padded to bucket 128). This pins, at those sizes, how much of the
evaluator callback is forward-compute (speedup-able by compile/TRT) vs fixed
marshalling (frombuffer/H2D/gather-softmax/D2H/tobytes -- NOT sped up). That
fraction sets the Amdahl ceiling for end-to-end self-play gains.
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
RESULT_JSON = Path(__file__).resolve().parent / "_results_callback_attr_small.json"


def build_payload(arr, n):
    import random
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn import rust_bridge

    rng = random.Random(7)
    states = []
    gi = 0
    while len(states) < n:
        st = engine.new_game(seed=200_000 + gi); gi += 1
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
    flats_rows = [list(int(i) for i in row) for row in payload["legal_flat_indices"]]
    offsets = [0]; flat = []
    for row in flats_rows:
        flat.extend(row); offsets.append(len(flat))
    return {
        "shape": shape, "inputs": bytes(payload["inputs"]),
        "legal_flat_indices_bytes": np.array(flat, dtype=np.int64).tobytes(),
        "legal_row_offsets": offsets,
    }


def wall(fn, iters=300):
    bc.cuda_sync(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    bc.cuda_sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from hexo_models.dense_cnn.inference import DenseCNNInference

    base = bc.build_training_model()
    inf = DenseCNNInference(base, device=DEVICE, amp=True, max_batch_size=1024, optimize_for_inference=True)
    arr = bc.generate_inputs(2048)

    out = []
    for n in (70, 99, 128, 256):
        payload = build_payload(arr, n)
        x = torch.from_numpy(np.ascontiguousarray(arr[:n])).to(DEVICE).to(memory_format=torch.channels_last)

        def raw():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return inf.model.forward_policy_value(x)

        def cb():
            return inf.evaluate_model1_payload(payload)

        bc.warmup(raw, seconds=4.0, min_iters=40)
        raw_ms = wall(raw)
        for _ in range(40):
            cb()
        cb_ms = wall(cb)
        rec = {"real_rows": n, "raw_forward_ms": raw_ms, "full_callback_ms": cb_ms,
               "forward_fraction_of_callback": raw_ms / cb_ms,
               "marshalling_ms": cb_ms - raw_ms}
        out.append(rec)
        print(f"[n={n}] raw_fwd={raw_ms:.4f}ms callback={cb_ms:.4f}ms "
              f"fwd_frac={rec['forward_fraction_of_callback']*100:.1f}% "
              f"marshalling={rec['marshalling_ms']:.4f}ms", flush=True)

    RESULT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
