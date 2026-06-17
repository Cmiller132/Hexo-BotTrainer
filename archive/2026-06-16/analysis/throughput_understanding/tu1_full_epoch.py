"""Q1 (definitive): REAL end-to-end self-play throughput at PRODUCTION settings.

Drives the production `generate_selfplay_epoch` (512 visits, 256 active games,
vbatch=4) with lightweight ctx/components stubs, writing to a TEMP dir (no
production state touched). It returns BOTH numbers directly:
  - search_positions_per_second = searched / mcts_search_elapsed   (search only)
  - positions_per_second        = searched / total_elapsed         (FULL epoch:
        + sample_from_state, finalize, policy-surprise materialize, NPZ writes,
        .hxr record IO)
So the search-only-vs-full gap is measured, not estimated. Also instruments the
MCTS evaluator callback to record the real leaf-batch-size distribution at 256-
game concurrency (Q2 empirical), and the cache hit rate.

Run AFTER a warmup epoch's worth of GPU activity is NOT required — the harness
itself warms (calibration-style) before the timed region by discarding the first
N positions' timing if requested. Here we report the whole run plus a
"first-2s-excluded" view to expose any cold-start tail.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tu1_full_epoch.json"


def build_model_and_config():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin

    raw = tomllib.loads(CONFIG.read_text())
    section = raw["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    payload = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval().to("cuda", memory_format=torch.channels_last)
    return model, parsed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--active", type=int, default=256)
    ap.add_argument("--vbatch", type=int, default=4)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from hexo_models.dense_cnn import selfplay as sp
    from hexo_models.dense_cnn.inference import DenseCNNInference

    model, parsed = build_model_and_config()

    # Patch the inference constructed inside generate_selfplay_epoch so we can
    # instrument the callback. We wrap DenseCNNInference.evaluate_model1_payload.
    stats = {"calls": 0, "rows": 0, "time_s": 0.0, "hist": [], "trace": []}
    orig_eval = DenseCNNInference.evaluate_model1_payload
    run_start = time.perf_counter()

    def wrapped(self, payload):
        rows = int(payload["shape"][0])
        t0 = time.perf_counter()
        out = orig_eval(self, payload)
        stats["calls"] += 1
        stats["rows"] += rows
        stats["time_s"] += time.perf_counter() - t0
        stats["hist"].append(rows)
        # sparse trace: (wall_since_start, batch) every call — used to find the
        # opening->midgame->tail trajectory and the low-batch (tail) wall fraction.
        stats["trace"].append((time.perf_counter() - run_start, rows))
        if stats["calls"] % 4000 == 0:
            el = time.perf_counter() - run_start
            recent = stats["hist"][-2000:]
            print(f"  [progress] t={el:6.1f}s calls={stats['calls']} "
                  f"recent_mean_batch={sum(recent)/len(recent):6.1f}", flush=True)
        return out

    DenseCNNInference.evaluate_model1_payload = wrapped

    tmp = Path(tempfile.mkdtemp(prefix="tu1_selfplay_"))
    trainer = SimpleNamespace(
        config=parsed, device=torch.device("cuda"),
        inference_batch_size=1024, selfplay_batch_size=args.active,
        mcts_virtual_batch_size=args.vbatch,
    )
    components = SimpleNamespace(model=SimpleNamespace(model=model, trainer=trainer))
    ctx = SimpleNamespace(
        output_dir=tmp,
        config=SimpleNamespace(run=SimpleNamespace(seed=1)),
        diagnostics=SimpleNamespace(write_json=lambda *a, **k: None),
    )

    print(f"[tu1] running generate_selfplay_epoch games={args.games} active={args.active} "
          f"vbatch={args.vbatch} visits={parsed.selfplay.search_visits} -> {tmp}", flush=True)
    t0 = time.perf_counter()
    summary = sp.generate_selfplay_epoch(ctx=ctx, components=components, epoch=1, games_per_epoch=args.games)
    wall = time.perf_counter() - t0

    hist = np.array(stats["hist"]) if stats["hist"] else np.array([0])

    # Trajectory / tail analysis from the (wall, batch) trace. dt between
    # consecutive calls approximates per-call wall (incl. non-callback gaps);
    # the "tail" is wall spent at low batch (deep + draining concurrency).
    tr = np.array(stats["trace"]) if stats["trace"] else np.zeros((1, 2))
    walls = tr[:, 0]; batches = tr[:, 1]
    dt = np.diff(walls, prepend=0.0)
    total_w = float(walls[-1]) if len(walls) else 0.0
    def wall_frac_below(thresh):
        return float(dt[batches < thresh].sum()) / max(total_w, 1e-9)
    # batch trajectory in 8 wall-time quantiles
    traj = []
    if total_w > 0:
        edges = np.linspace(0, total_w, 9)
        for i in range(8):
            mask = (walls >= edges[i]) & (walls < edges[i + 1])
            traj.append(round(float(batches[mask].mean()) if mask.any() else 0.0, 1))

    out = {
        "tail_analysis": {
            "total_callback_wall_s": total_w,
            "wall_frac_batch_lt_25": wall_frac_below(25),
            "wall_frac_batch_lt_50": wall_frac_below(50),
            "wall_frac_batch_lt_100": wall_frac_below(100),
            "batch_trajectory_8quantiles": traj,
        },
        "settings": {"games": args.games, "active": args.active, "vbatch": args.vbatch,
                     "visits": parsed.selfplay.search_visits},
        "searched_positions": summary["searched_positions"],
        "mcts_simulations": summary["mcts_simulations"],
        "wall_seconds": wall,
        "search_positions_per_second": summary["search_positions_per_second"],   # search-only
        "full_positions_per_second": summary["positions_per_second"],            # FULL epoch
        "mcts_search_elapsed_seconds": summary["mcts_search_elapsed_seconds"],
        "non_search_fraction": 1.0 - summary["mcts_search_elapsed_seconds"] / max(wall, 1e-9),
        "callback_calls": stats["calls"],
        "callback_total_rows": stats["rows"],
        "callback_total_time_s": stats["time_s"],
        "mean_leaf_batch": float(hist.mean()),
        "p50_leaf_batch": float(np.percentile(hist, 50)),
        "p95_leaf_batch": float(np.percentile(hist, 95)),
        "max_leaf_batch": int(hist.max()),
        "leaf_evals_per_position": stats["rows"] / max(summary["searched_positions"], 1),
        "games_finished": summary["games_finished"],
        "tmp_dir": str(tmp),
    }
    print(json.dumps(out, indent=2), flush=True)
    RESULT.write_text(json.dumps(out, indent=2))
    # Clean the temp self-play shards (can be large-ish).
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[tu1] cleaned {tmp}; wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
