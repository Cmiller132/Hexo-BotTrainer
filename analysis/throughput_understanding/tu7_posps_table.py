"""Measured 4-config self-play pos/s table (real full pipeline, WSL).

Runs generate_selfplay_epoch (full pipeline: search + sample creation + NPZ
writes) at IDENTICAL settings/seed for four forward configs, toggled via the
env gates added to DenseCNNInference:
  1. baseline   : FP16 torch, power-of-two bucketing (current production)
  2. +bucketing : FP16 torch, multiple-of-16 bucketing (the padding fix)
  3. +trt       : TensorRT FP16 forward, power-of-two bucketing
  4. +trt+bucket: TensorRT FP16 forward, multiple-of-16 bucketing

Same seed => same games => clean relative comparison. Reports full pos/s,
search-only pos/s, and whether TRT was actually adopted (correctness gate).
Each config builds its own evaluator (TRT engine rebuilt per epoch, as in prod).
Run in WSL: python analysis/throughput_understanding/tu7_posps_table.py --games 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"
RESULT = Path(__file__).resolve().parent / "_tu7_posps_table.json"

CONFIGS = [
    ("baseline",    {"HEXO_TRT": None, "HEXO_BUCKET_PAD_MULTIPLE": None}),
    ("+bucketing",  {"HEXO_TRT": None, "HEXO_BUCKET_PAD_MULTIPLE": "16"}),
    ("+trt",        {"HEXO_TRT": "1",  "HEXO_BUCKET_PAD_MULTIPLE": None}),
    ("+trt+bucket", {"HEXO_TRT": "1",  "HEXO_BUCKET_PAD_MULTIPLE": "16"}),
]


def build():
    import tomllib
    from hexo_models.dense_cnn.config import parse_model1_config
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.loads(CONFIG.read_text())["model"]["config"]
    parsed = parse_model1_config(section)
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    model.eval().to("cuda", memory_format=torch.channels_last)
    return model, parsed


def set_env(d):
    for k, v in d.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_one(name, model, parsed, games, active, vbatch):
    from hexo_models.dense_cnn import selfplay as sp
    from hexo_models.dense_cnn.inference import DenseCNNInference

    # Capture whether TRT was adopted by patching to read trt_info post-build.
    captured = {}
    orig_init = DenseCNNInference.__init__
    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        captured["trt_info"] = getattr(self, "trt_info", None)
        captured["bucket_pad_multiple"] = getattr(self, "bucket_pad_multiple", None)
    DenseCNNInference.__init__ = patched_init
    try:
        tmp = Path(tempfile.mkdtemp(prefix=f"tu7_{name}_"))
        trainer = SimpleNamespace(config=parsed, device=torch.device("cuda"),
                                  inference_batch_size=1024, selfplay_batch_size=active,
                                  mcts_virtual_batch_size=vbatch)
        components = SimpleNamespace(model=SimpleNamespace(model=model, trainer=trainer))
        ctx = SimpleNamespace(output_dir=tmp,
                              config=SimpleNamespace(run=SimpleNamespace(seed=1)),
                              diagnostics=SimpleNamespace(write_json=lambda *a, **k: None))
        t0 = time.perf_counter()
        summary = sp.generate_selfplay_epoch(ctx=ctx, components=components, epoch=1, games_per_epoch=games)
        wall = time.perf_counter() - t0
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
        return {
            "config": name,
            "full_pos_per_s": summary["positions_per_second"],
            "search_pos_per_s": summary["search_positions_per_second"],
            "searched_positions": summary["searched_positions"],
            "wall_s": wall,
            "trt_info": captured.get("trt_info"),
            "bucket_pad_multiple": captured.get("bucket_pad_multiple"),
        }
    finally:
        DenseCNNInference.__init__ = orig_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--active", type=int, default=96)
    ap.add_argument("--vbatch", type=int, default=4)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)

    model, parsed = build()
    results = []
    for name, env in CONFIGS:
        set_env(env)
        print(f"\n=== config {name} (env {env}) ===", flush=True)
        r = run_one(name, model, parsed, args.games, args.active, args.vbatch)
        results.append(r)
        ti = r["trt_info"] or {}
        print(f"  full_pos/s={r['full_pos_per_s']:.2f} search_pos/s={r['search_pos_per_s']:.2f} "
              f"trt_adopted={ti.get('adopted')} bucket_mult={r['bucket_pad_multiple']}", flush=True)
        set_env({k: None for k in env})

    base = results[0]["full_pos_per_s"]
    print("\n=== pos/s table (full pipeline) ===", flush=True)
    for r in results:
        print(f"  {r['config']:>12s}: {r['full_pos_per_s']:7.2f} pos/s "
              f"({r['full_pos_per_s']/base:.2f}x baseline)", flush=True)
    RESULT.write_text(json.dumps({"settings": vars(args), "results": results}, indent=2))
    print(f"[tu7] wrote {RESULT.name}", flush=True)


if __name__ == "__main__":
    main()
