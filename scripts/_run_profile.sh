#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer
VENV="${VENV:-/root/.venvs/hexo-bottrainer-wsl}"
PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export HEXO_TRT=1
export HEXO_MCTS_TRACE=1
export HEXO_BUCKET_PAD_MULTIPLE=16
OUT="$ROOT/runs/profile_selfplay.out"
echo "=== profiling $(date -u +%FT%TZ) py=$PY ===" | tee "$OUT"
"$PY" "$ROOT/scripts/_profile_selfplay.py" >>"$OUT" 2>&1
echo "=== exit rc=$? ===" | tee -a "$OUT"
