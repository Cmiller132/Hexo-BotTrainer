#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
python scripts/_perf_tss_ab2.py --games 64 --moves 5 --visits 512 --vbatch 64 --tag SPLIT 2>&1 | tail -6
echo "ALL_DONE"
