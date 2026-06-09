#!/bin/bash
# Restore the canonical 8080 dashboard on the MAIN tree (read-only on runs/), as a
# harness-managed foreground process (NOT setsid -> won't get reaped). Live venv,
# cwd = main tree so it scans the main tree's runs/ (dense_cnn + the hexgt mirror).
# Does not touch the GPU or the running sweep.
set -uo pipefail
MAIN=/mnt/e/Hexo-BotTrainer
VENV=/root/.venvs/hexo-bottrainer-wsl
cd "$MAIN" || exit 11
source "$VENV/bin/activate" 2>/dev/null || true
export PYTHONPATH="$MAIN/packages/hexo_engine/python:$MAIN/packages/hexo_utils/python:$MAIN/packages/hexo_runner/python:$MAIN/packages/hexo_train/python:$MAIN/packages/hexo_models/python:$MAIN/packages/hexo_frontend/python"
echo "starting 8080 dashboard from $MAIN ($(date +%H:%M:%S))"
exec python -m hexo_frontend.web --host 0.0.0.0 --port 8080
