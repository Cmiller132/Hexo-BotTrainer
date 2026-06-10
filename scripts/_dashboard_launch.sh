#!/usr/bin/env bash
# Launch the :8080 Flask dashboard (hexo_frontend.web) detached. It scans
# cwd/runs, so cwd MUST be /mnt/e/Hexo-BotTrainer (the run mount root, NOT the
# -hexgt repo). Single-instance: no-op if one is already listening.
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
RUNROOT=/mnt/e/Hexo-BotTrainer
PORT="${1:-8080}"

if pgrep -f "hexo_frontend.web" >/dev/null 2>&1; then
  echo "dashboard already running: $(pgrep -f hexo_frontend.web | tr '\n' ' ')"
  exit 0
fi

export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python:$ROOT/packages/dense_cnn_restnet/python"
export HEXO_DEBUG_RUN_ROOT="$RUNROOT"
cd "$RUNROOT"
LOG="$RUNROOT/dashboard.out.log"
setsid "$VENV/bin/python" -u -m hexo_frontend.web --host 0.0.0.0 --port "$PORT" \
  --sealbot-path /mnt/e/SealBot >"$LOG" 2>&1 &
echo "launched dashboard pid=$! port=$PORT log=$LOG"
sleep 4
echo "--- log tail ---"; tail -12 "$LOG" 2>/dev/null
echo "--- listening? ---"; ss -ltn 2>/dev/null | grep ":$PORT" || echo "NOT yet listening on :$PORT"
