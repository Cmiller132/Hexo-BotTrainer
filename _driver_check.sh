#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python - <<'PY'
import argparse, inspect
# 1) argparse parses with the new exploration flags
import runpy, sys
src = open("scripts/_rl_train.py").read()
assert "--total-alpha" in src and "--root-policy-temperature" in src and "--eval-opening-moves" in src
# 2) cfg accepts the new selfplay keys
from hexo_models.hexgt.config import parse_hexgt_config
cfg = parse_hexgt_config({"selfplay": {"root_dirichlet_total_alpha": 6.6, "root_dirichlet_noise_fraction": 0.25,
        "root_policy_temperature": 1.0, "c_puct": 1.5, "temperature_floor": 0.1, "forced_playout_k": 2.0}})
assert abs(cfg.selfplay.root_dirichlet_total_alpha - 6.6) < 1e-9
# 3) make_hexgt_factory accepts opening args
from hexo_models.hexgt.evaluation import make_hexgt_factory
sig = inspect.signature(make_hexgt_factory)
assert "opening_moves" in sig.parameters and "opening_temperature" in sig.parameters
# 4) SelfPlayResult exposes the Q-metric attributes the log line uses
from hexo_models.hexgt.selfplay import SelfPlayResult
for f in ("decisive_fraction","game_length_median","opening_unique_fraction","move2_entropy",
          "mean_abs_value","draw_fraction","forced_move_fraction"):
    assert hasattr(SelfPlayResult, f) or f in SelfPlayResult.__dataclass_fields__, f
print("DRIVER WIRING OK: argparse flags, cfg keys, eval opening args, Q-metric fields all present")
PY
echo "EXIT=$?"
