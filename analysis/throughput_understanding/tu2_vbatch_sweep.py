"""Q4 (headroom) + Q1 (cold vs warm): sweep virtual_batch_size in real self-play.

Uses the production search-only probe (`_benchmark_selfplay_setting`, position-
capped, 256 active games, 512 visits) at vbatch in {1,4,8,16,32,64}. For each:
pos/s, mean/p50/p95 leaf-batch, leaf-evals/position, cache hit rate, callback
fwd/s. Answers: does aggregating more leaves per round (fatter forwards) raise
throughput, or is ~99 already near the efficiency knee?

Also does a COLD probe (vbatch=4, as the first self-play activity, only the
DenseCNNInference 8-iter warmup) vs the WARM sweep value — to reconcile the
calibration 12.8 vs the warm 58.8 (same code path, different GPU thermal/clock +
cuDNN autotune state).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tu2_vbatch_sweep.json"


def build():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin

    raw = tomllib.loads(CONFIG.read_text())
    section = raw["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval()
    return model, parsed


def gpu_clock():
    import subprocess
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()[0])
    except Exception:
        return -1


def run_probe(inf, parsed, vbatch, positions, stats):
    from hexo_models.dense_cnn import performance as perf
    stats["calls"] = 0; stats["rows"] = 0; stats["time_s"] = 0.0; stats["hist"] = []
    t0 = time.perf_counter()
    res = perf._benchmark_selfplay_setting(
        inference=inf, config=parsed,
        selfplay_batch_size=parsed.selfplay.active_games,
        virtual_batch_size=vbatch, visits=parsed.selfplay.search_visits,
        probe_positions=positions,
    )
    wall = time.perf_counter() - t0
    hist = np.array(stats["hist"]) if stats["hist"] else np.array([0])
    diag = res.get("mcts_diagnostics", {})
    req = diag.get("eval_requested_states", 0); uniq = diag.get("eval_unique_states", 0)
    return {
        "vbatch": vbatch, "positions": res["positions"],
        "pos_per_s": res["positions_per_second"], "wall_s": wall,
        "mean_leaf_batch": float(hist.mean()), "p50_leaf_batch": float(np.percentile(hist, 50)),
        "p95_leaf_batch": float(np.percentile(hist, 95)), "max_leaf_batch": int(hist.max()),
        "callback_calls": stats["calls"], "callback_rows": stats["rows"],
        "callback_fwd_per_s": stats["rows"] / max(stats["time_s"], 1e-9),
        "leaf_evals_per_position": stats["rows"] / max(res["positions"], 1),
        "eval_requested_states": req, "eval_unique_states": uniq,
        "cache_hit_frac": 1.0 - uniq / max(req, 1),
        "clock_mhz": gpu_clock(),
    }


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from hexo_models.dense_cnn.inference import DenseCNNInference

    model, parsed = build()
    inf = DenseCNNInference(model, device="cuda", amp=parsed.training.amp,
                            return_logits=False, max_batch_size=1024)

    stats = {}
    orig = inf.evaluate_model1_payload
    def wrapped(payload):
        rows = int(payload["shape"][0]); t0 = time.perf_counter()
        out = orig(payload)
        stats["calls"] += 1; stats["rows"] += rows; stats["time_s"] += time.perf_counter() - t0
        stats["hist"].append(rows)
        return out
    inf.evaluate_model1_payload = wrapped

    out = {"cold": None, "sweep": []}

    # COLD: first real self-play activity (only the inference __init__ 8-iter warmup).
    print("[tu2] COLD probe (vbatch=4, first self-play)...", flush=True)
    out["cold"] = run_probe(inf, parsed, 4, 2048, stats)
    print(f"  cold pos/s={out['cold']['pos_per_s']:.1f} clock={out['cold']['clock_mhz']}MHz "
          f"mean_batch={out['cold']['mean_leaf_batch']:.1f}", flush=True)

    # WARM-UP throwaway.
    print("[tu2] warm-up...", flush=True)
    run_probe(inf, parsed, 4, 1024, stats)

    for vbatch in (1, 4, 8, 16, 32, 64):
        r = run_probe(inf, parsed, vbatch, 2048, stats)
        out["sweep"].append(r)
        print(f"  vbatch={vbatch:>2d} pos/s={r['pos_per_s']:6.1f} mean_batch={r['mean_leaf_batch']:6.1f} "
              f"p95={r['p95_leaf_batch']:6.1f} evals/pos={r['leaf_evals_per_position']:5.1f} "
              f"cache_hit={r['cache_hit_frac']*100:4.1f}% cb_fwd/s={r['callback_fwd_per_s']:7.0f} "
              f"clk={r['clock_mhz']}", flush=True)

    RESULT.write_text(json.dumps(out, indent=2))
    print(f"[tu2] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
