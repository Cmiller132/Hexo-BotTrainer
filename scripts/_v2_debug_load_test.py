"""One-shot: verify the dashboard debug loader handles a v2+raytap checkpoint.

Exercises the exact worker path: _hexfield_eq(meta) env seeding ->
infer_net_kwargs -> strict load -> featurize (46-plane v2 + raylen for
ray-taps) -> CPU forward. Run in the WSL hexgt-build venv, CPU only.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("hexo_frontend", "hexo_engine", "hexo_train"):
    p = ROOT / "packages" / pkg / "python"
    if p.is_dir():
        sys.path.insert(0, str(p))

import torch

from hexo_frontend import debug_infer as di

CKPT = Path(sys.argv[1] if len(sys.argv) > 1
            else "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main2_prefit/a5/soak_init.pt")
EXPECT_NF = int(sys.argv[2]) if len(sys.argv) > 2 else 46
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
meta = payload["meta"]
print("meta feature_version/width/raytap/trunk:",
      meta.get("feature_version"), meta.get("feature_width"),
      meta.get("raytap"), meta.get("trunk_layout"))

loaded = di._load_hexfield_eq_checkpoint(CKPT, payload)
print("load_warnings:", loaded.load_warnings)
assert not loaded.load_warnings, loaded.load_warnings

eq = di._hexfield_eq()
print("imported NUM_FEATURES:", eq.constants.NUM_FEATURES,
      "support_radius:", eq.support_radius)
assert eq.constants.NUM_FEATURES == EXPECT_NF, (eq.constants.NUM_FEATURES, EXPECT_NF)

# The full Debug-tab analyze path on the empty board (featurize v2 46 planes,
# build raylen for the ray-taps, CPU forward, decode heads).
out = di._analyze_hexfield_eq(loaded, [])
top = out["policy"][:3] if isinstance(out.get("policy"), list) else None
print("analyze keys:", sorted(out.keys())[:10])
print("value:", out.get("value"), "top3:", top)
print("OK: v2+raytap checkpoint loads and analyzes through the debug path")
