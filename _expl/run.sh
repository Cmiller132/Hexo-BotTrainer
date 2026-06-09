#!/usr/bin/env bash
# Read-only CPU runner for exploration analysis. Usage: run.sh <python-file> [args...]
set -uo pipefail
R=/mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python:$R/packages/hexgnn/python:$R/packages/hexo_frontend/python"
exec /root/.venvs/hexgt-build/bin/python -u "$@"
