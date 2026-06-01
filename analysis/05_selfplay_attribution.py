"""End-to-end self-play attribution: where does the move-time actually go?

Runs the production self-play loop (Rust MCTS, visits=512, active_games=256,
virtual_batch=4) via performance._benchmark_selfplay_setting with an INSTRUMENTED
DenseCNNInference whose evaluate_model1_payload is wrapped to record: call count,
rows per call (real leaf batch size), and forward(callback) wall time.

Answers: real pos/s, real mean leaf-batch size, and what fraction of self-play
wall time is the evaluator callback (== the thing bf16/compile/TRT speed up).
This sets the ceiling on end-to-end gains from forward optimization.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_common as bc

DEVICE = "cuda"
RESULT_JSON = Path(__file__).resolve().parent / "_results_selfplay_attribution.json"


def main():
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn import performance as perf

    cfg = parse_model1_config(bc._load_model_config())
    model = bc.build_training_model()

    inf = DenseCNNInference(
        model, device=DEVICE, amp=cfg.training.amp, return_logits=False, max_batch_size=1024,
    )

    # Instrument the real MCTS callback.
    stats = {"calls": 0, "rows": 0, "time_s": 0.0, "batch_hist": []}
    orig = inf.evaluate_model1_payload

    def wrapped(payload):
        rows = int(payload["shape"][0])
        t0 = time.perf_counter()
        out = orig(payload)
        dt = time.perf_counter() - t0
        stats["calls"] += 1
        stats["rows"] += rows
        stats["time_s"] += dt
        stats["batch_hist"].append(rows)
        return out

    inf.evaluate_model1_payload = wrapped

    target = 2048
    wall0 = time.perf_counter()
    res = perf._benchmark_selfplay_setting(
        inference=inf, config=cfg,
        selfplay_batch_size=cfg.selfplay.active_games,   # 256
        virtual_batch_size=4,                            # production calibrated value
        visits=cfg.selfplay.search_visits,               # 512
        probe_positions=target,
    )
    wall = time.perf_counter() - wall0

    hist = np.array(stats["batch_hist"])
    out = {
        "positions": res["positions"],
        "positions_per_second": res["positions_per_second"],
        "wall_seconds": wall,
        "callback_calls": stats["calls"],
        "callback_total_rows": stats["rows"],
        "callback_total_time_s": stats["time_s"],
        "callback_fraction_of_wall": stats["time_s"] / wall,
        "mean_leaf_batch": float(hist.mean()),
        "p50_leaf_batch": float(np.percentile(hist, 50)),
        "p95_leaf_batch": float(np.percentile(hist, 95)),
        "max_leaf_batch": int(hist.max()),
        "forwards_per_s_through_net": stats["rows"] / max(stats["time_s"], 1e-9),
        "leaf_evals_per_position": stats["rows"] / max(res["positions"], 1),
        "mcts_diag": res.get("mcts_diagnostics", {}),
    }
    print(json.dumps(out, indent=2), flush=True)
    RESULT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {RESULT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
