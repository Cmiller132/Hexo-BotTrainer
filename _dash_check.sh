#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
cd "$ROOT" || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
echo "=== hexo_frontend importable? ==="
python -c "import hexo_frontend.web as w; print('OK import', w.__file__)" 2>&1 | tail -2
echo "=== eval JSONs present so far ==="
ls "$ROOT/runs/hexgt_rl_main/eval/"*.json 2>/dev/null
echo "=== run bridge once ==="
python _dashboard_bridge.py --once 2>&1 | tail -3
echo "=== diagnostics produced ==="
ls "$ROOT/runs/hexgt_rl_main/diagnostics/" 2>/dev/null
