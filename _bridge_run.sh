#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt; VENV=/root/.venvs/hexgt-build
cd "$ROOT" || exit 11
pkill -9 -f "_dashboard_bridge.py" 2>/dev/null; sleep 1
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
echo "bridge loop starting (foreground, harness-kept-alive) $(date +%H:%M:%S)"
exec python -u "$ROOT/_dashboard_bridge.py"
