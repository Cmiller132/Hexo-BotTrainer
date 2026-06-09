#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export PYTHONFAULTHANDLER=1
cd "$ROOT"
python - <<'PY'
import faulthandler, sys
faulthandler.enable()
import hexo_engine, hexo_models
print("hexo_engine pkg :", hexo_engine.__file__)
import hexo_engine._rust as er
print("hexo_engine._rust:", er.__file__)
import hexo_models._rust as mr
print("hexo_models._rust:", mr.__file__)
import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
s = eng.new_game()
for q, r in [(0,0),(0,7),(2,7),(1,0),(4,0),(4,7),(6,7),(5,0),(0,-2),(0,8),(2,8)]:
    eng.apply_action(s, PlacementAction(AxialCoord(q=q, r=r)))
print("state built; phase:", eng.to_python_state(s).phase.name)
from hexo_models.hexgt import rust_bridge as rb
print("hexgt module:", rb._hexgt_module().__file__ if hasattr(rb._hexgt_module(), "__file__") else "(builtin)")
print("calling hexgt_threat_analysis ...")
a = rb._hexgt_module().hexgt_threat_analysis(s)
print("OK ->", dict(a))
PY
echo "=== rc=$? ==="
