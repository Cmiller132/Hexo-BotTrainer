"""Tail quantification core: pos/s as a function of ACTIVE-GAME concurrency,
plus the adaptive-virtual_batch_size rescue.

As a fixed cohort of games drains, active concurrency falls 256 -> 1. This sweeps
the production search probe at fixed concurrency levels (vbatch=4) to MEASURE
pos/s, leaf-batch, evals/position, and cache-hit at each level -> the throughput
penalty of the low-concurrency tail.

Then re-runs the low levels with ADAPTIVE vbatch = max(4, round(256*4/active))
(hold the per-round leaf-request budget ~constant at the 256-game value) to
MEASURE how much of the tail penalty an adaptive-vbatch policy recovers.
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
RESULT = Path(__file__).resolve().parent / "_tu5_concurrency.json"


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


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn import performance as perf

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

    def probe(active, vbatch, positions=1024):
        stats["calls"] = 0; stats["rows"] = 0; stats["time_s"] = 0.0; stats["hist"] = []
        res = perf._benchmark_selfplay_setting(
            inference=inf, config=parsed,
            selfplay_batch_size=active, virtual_batch_size=vbatch,
            visits=parsed.selfplay.search_visits, probe_positions=positions)
        h = np.array(stats["hist"]) if stats["hist"] else np.array([0])
        diag = res.get("mcts_diagnostics", {})
        req = diag.get("eval_requested_states", 0); uniq = diag.get("eval_unique_states", 0)
        return {"active": active, "vbatch": vbatch, "pos_per_s": res["positions_per_second"],
                "mean_batch": float(h.mean()), "p95_batch": float(np.percentile(h, 95)),
                "evals_per_pos": stats["rows"] / max(res["positions"], 1),
                "cache_hit": 1.0 - uniq / max(req, 1),
                "callback_fwd_per_s": stats["rows"] / max(stats["time_s"], 1e-9)}

    # warm-up
    probe(256, 4, 768)

    out = {"fixed_vbatch4": [], "adaptive_vbatch": []}
    print("[tu5] concurrency sweep (fixed vbatch=4):", flush=True)
    for active in (256, 192, 128, 96, 64, 48, 32, 16, 8, 4, 2, 1):
        r = probe(active, 4)
        out["fixed_vbatch4"].append(r)
        print(f"  active={active:>3d} pos/s={r['pos_per_s']:6.1f} mean_batch={r['mean_batch']:6.1f} "
              f"evals/pos={r['evals_per_pos']:5.1f} cache_hit={r['cache_hit']*100:4.1f}% "
              f"cb_fwd/s={r['callback_fwd_per_s']:6.0f}", flush=True)

    print("[tu5] adaptive vbatch = max(4, round(1024/active)):", flush=True)
    for active in (128, 64, 32, 16, 8, 4):
        vb = max(4, round(1024 / active))
        r = probe(active, vb)
        out["adaptive_vbatch"].append(r)
        print(f"  active={active:>3d} vbatch={vb:>4d} pos/s={r['pos_per_s']:6.1f} "
              f"mean_batch={r['mean_batch']:6.1f} evals/pos={r['evals_per_pos']:5.1f} "
              f"cache_hit={r['cache_hit']*100:4.1f}%", flush=True)

    RESULT.write_text(json.dumps(out, indent=2))
    print(f"[tu5] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
