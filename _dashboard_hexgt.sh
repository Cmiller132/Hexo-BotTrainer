#!/usr/bin/env bash
# Launch a SECOND, read-only dashboard for the hexgt worktree on port 8081, plus
# the diagnostics bridge loop. Both setsid-detached so they persist independently
# (the live RL run keeps the WSL VM alive). Does NOT touch the main :8080 dashboard
# or the RL job (the bridge only writes to the run's diagnostics/ subdir, which the
# RL job never reads/writes; the dashboard is GPU-free and read-only on runs/).
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
PP="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
LOGDIR="$ROOT/runs/hexgt_rl_main"
mkdir -p "$LOGDIR"

# Bridge loop: eval/*.json -> diagnostics/dense_cnn.*.json (read-only on RL files).
setsid bash -c "cd '$ROOT'; source '$VENV/bin/activate'; export PYTHONPATH='$PP'; exec python '$ROOT/_dashboard_bridge.py'" \
  >> "$LOGDIR/_bridge.log" 2>&1 &
echo "bridge launched (pid $!)"

# Dashboard on :8081 (cwd=worktree root so it scans the worktree's runs/).
setsid bash -c "cd '$ROOT'; source '$VENV/bin/activate'; export PYTHONPATH='$PP'; exec python -m hexo_frontend.web --host 0.0.0.0 --port 8081" \
  >> "$LOGDIR/_dashboard8081.log" 2>&1 &
echo "dashboard launched (pid $!)"
sleep 5
echo "--- bridge log ---"; tail -2 "$LOGDIR/_bridge.log" 2>/dev/null
echo "--- dashboard log ---"; tail -5 "$LOGDIR/_dashboard8081.log" 2>/dev/null
