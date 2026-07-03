#!/usr/bin/env bash
# Training-step A/B matrix (run in WSL, GPU free).
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
PY=/root/.venvs/hexgt-build/bin/python

echo '=== eager + train_flex (production) ==='
HEXFIELD_TRAIN_FLEX=1 "$PY" scripts/_hexfield_train_step_bench.py shapes
echo '=== eager + train_flex_pair ==='
HEXFIELD_TRAIN_FLEX=1 HEXFIELD_TRAIN_FLEX_PAIR=1 "$PY" scripts/_hexfield_train_step_bench.py shapes
echo '=== eager materialized (_BiasGather) ==='
"$PY" scripts/_hexfield_train_step_bench.py
echo '=== compiled + train_flex ==='
HEXFIELD_TRAIN_FLEX=1 "$PY" scripts/_hexfield_train_step_bench.py compiled shapes
echo '=== compiled + train_flex_pair ==='
HEXFIELD_TRAIN_FLEX=1 HEXFIELD_TRAIN_FLEX_PAIR=1 "$PY" scripts/_hexfield_train_step_bench.py compiled shapes
