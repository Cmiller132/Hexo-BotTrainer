#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HEXO_MCTS_TRACE=1
cd "$ROOT"
echo "### traced self-play (fastpath ON), 6 moves x 64 games, visits=512 ###"
python scripts/_perf_tss_ab2.py --games 64 --moves 6 --visits 512 --vbatch 64 --tag TRACE 2>&1 | grep -E "mcts-trace|TRACE\]|pos/s" | tail -20
echo "ALL_DONE"
