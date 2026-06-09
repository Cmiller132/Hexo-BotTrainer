#!/usr/bin/env bash
# CPU-only dev/test env for the hexgnn lineage. Sets PYTHONPATH to the local
# worktree packages (so a stale installed wheel is never imported) PLUS the new
# top-level hexgnn package, hides the GPU (the live hexgt_rl_main3 run owns it),
# and runs whatever command is passed. Usage: bash _hexgnn_env.sh <cmd...>
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
PP="$ROOT/packages/hexo_engine/python"
PP="$PP:$ROOT/packages/hexo_utils/python"
PP="$PP:$ROOT/packages/hexo_runner/python"
PP="$PP:$ROOT/packages/hexo_train/python"
PP="$PP:$ROOT/packages/hexo_models/python"
PP="$PP:$ROOT/packages/hexgnn/python"
export PYTHONPATH="$PP"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$VENV/bin/python" "$@"
