#!/usr/bin/env bash
# Durable dashboard bridge (harness-managed foreground; NOT setsid). Reads the
# worktree RL run, writes MAIN-tree diagnostics (buffer + length stats, games).
set -uo pipefail
WT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
cd "$WT"
source "$VENV/bin/activate" 2>/dev/null || true
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_train/python:$WT/packages/hexo_models/python:$WT/packages/hexo_frontend/python"
echo "starting bridge ($(date +%H:%M:%S))"
exec python _dashboard_bridge.py
