"""CPU probe: the live main_4 reference checkpoint strict-loads via the new
evaluation._load_reference_network (READ-ONLY on the runs dir).

Validates the activation requirement for the live run: main_3 ckpt10 is
architecture-identical to main_4 and its payload carries model_state.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for pkg in ("hexo_models", "hexo_engine", "hexo_runner", "hexo_train", "hexo_utils"):
    p = ROOT / "packages" / pkg / "python"
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(ROOT / "packages" / "dense_cnn_restnet" / "python"))

from dense_cnn_restnet.config import parse_model1_config  # noqa: E402
from dense_cnn_restnet.evaluation import _load_reference_network  # noqa: E402

raw = tomllib.loads((ROOT / "configs" / "dense_cnn_restnet_main_4.toml").read_text(encoding="utf-8"))
cfg = parse_model1_config(raw["model"]["config"])
print(f"reference_checkpoint = {cfg.evaluation.reference_checkpoint}")
print(f"reference_games      = {cfg.evaluation.reference_games}")
print(f"games_per_epoch      = {cfg.evaluation.games_per_epoch}")

net = _load_reference_network(cfg.evaluation.reference_checkpoint, cfg.architecture)
params = sum(p.numel() for p in net.parameters())
print(f"strict load OK: RestnetNetwork params={params:,} training={net.training}")
