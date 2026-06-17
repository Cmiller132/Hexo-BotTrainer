"""Per-config pos/s at PRODUCTION concurrency (256), bounded search-only probe.

Full-epoch runs at 256 games x 512 visits are ~20-30 min each (infeasible x5),
so this uses the position-capped search probe (_benchmark_selfplay_setting) at
active=256 to compare forward configs at the production concurrency, at a
moderate depth (probe_positions chosen for ~12 moves). Search-only; multiply by
the measured full-pipeline factor (~0.93, from the 6-game full-epoch smoke) for
end-to-end. Adaptive-vbatch is a no-op here (constant concurrency) — its benefit
is the drain tail, measured separately (tu5 / full-epoch).

Configs via the inference env gates: baseline / +bucketing(16) / +TRT / +both.
Run in WSL (TRT): python analysis/throughput_understanding/tu8_config_posps.py
"""

from __future__ import annotations

import json
import os
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
RESULT = Path(__file__).resolve().parent / "_tu8_config_posps.json"

_KEYS = ("HEXO_TRT", "HEXO_BUCKET_PAD_MULTIPLE")
CONFIGS = [
    ("baseline",   {}),
    ("+bucketing", {"HEXO_BUCKET_PAD_MULTIPLE": "16"}),
    ("+trt",       {"HEXO_TRT": "1"}),
    ("+combined",  {"HEXO_TRT": "1", "HEXO_BUCKET_PAD_MULTIPLE": "16"}),
]


def build():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.loads(CONFIG.read_text())["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval()
    return model, parsed


def set_env(d):
    for k, v in d.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=int, default=256)
    ap.add_argument("--positions", type=int, default=3072)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    from hexo_models.dense_cnn.inference import DenseCNNInference
    from hexo_models.dense_cnn import performance as perf

    model, parsed = build()
    results = []
    for name, env in CONFIGS:
        set_env({k: None for k in _KEYS})
        set_env(env)
        # Fresh evaluator per config so the TRT engine / bucketing gate is rebuilt.
        inf = DenseCNNInference(model, device="cuda", amp=parsed.training.amp,
                                return_logits=False, max_batch_size=1024)
        trt_info = getattr(inf, "trt_info", None)
        stats = {"calls": 0, "rows": 0, "time_s": 0.0, "hist": []}
        orig = inf.evaluate_model1_payload
        def wrapped(payload, _orig=orig, _s=stats):
            rows = int(payload["shape"][0]); t0 = time.perf_counter()
            out = _orig(payload)
            _s["calls"] += 1; _s["rows"] += rows; _s["time_s"] += time.perf_counter() - t0
            _s["hist"].append(rows)
            return out
        inf.evaluate_model1_payload = wrapped
        # warm
        perf._benchmark_selfplay_setting(inference=inf, config=parsed,
            selfplay_batch_size=args.active, virtual_batch_size=4,
            visits=parsed.selfplay.search_visits, probe_positions=max(512, args.positions // 3))
        stats.update(calls=0, rows=0, time_s=0.0); stats["hist"] = []
        res = perf._benchmark_selfplay_setting(inference=inf, config=parsed,
            selfplay_batch_size=args.active, virtual_batch_size=4,
            visits=parsed.selfplay.search_visits, probe_positions=args.positions)
        h = np.array(stats["hist"]) if stats["hist"] else np.array([0])
        r = {"config": name, "search_pos_per_s": res["positions_per_second"],
             "est_full_pos_per_s": res["positions_per_second"] * 0.93,
             "mean_batch": float(h.mean()), "callback_fwd_per_s": stats["rows"]/max(stats["time_s"],1e-9),
             "trt_adopted": (trt_info or {}).get("adopted"),
             "trt_value_err": (trt_info or {}).get("value_max_abs_err"),
             "trt_argmax_match": (trt_info or {}).get("policy_argmax_match"),
             "bucket_pad_multiple": getattr(inf, "bucket_pad_multiple", None)}
        results.append(r)
        print(f"  {name:>12s}: search_pos/s={r['search_pos_per_s']:6.1f} "
              f"(~full {r['est_full_pos_per_s']:6.1f}) mean_batch={r['mean_batch']:6.1f} "
              f"trt={r['trt_adopted']} cb_fwd/s={r['callback_fwd_per_s']:.0f}", flush=True)
        del inf
    set_env({k: None for k in _KEYS})

    base = results[0]["search_pos_per_s"]
    print("\n=== per-config pos/s @ active=256 (search-only; ~full = x0.93) ===", flush=True)
    for r in results:
        print(f"  {r['config']:>12s}: {r['search_pos_per_s']:6.1f} ({r['search_pos_per_s']/base:.2f}x)", flush=True)
    RESULT.write_text(json.dumps({"settings": vars(args), "results": results}, indent=2))
    print(f"[tu8] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
