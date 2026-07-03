#!/usr/bin/env bash
# Training matrix v2: static-Npad compile (multi-shape) + raised pair budget.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128
export MALLOC_TRIM_THRESHOLD_=536870912 MALLOC_MMAP_THRESHOLD_=536870912
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
PY=/root/.venvs/hexgt-build/bin/python

echo '=== eager + train_flex, big budget (8e7) ==='
HEXFIELD_TRAIN_FLEX=1 "$PY" scripts/_hexfield_train_step_bench.py big
echo '=== compiled(static-npad) + train_flex, multi-shape ==='
HEXFIELD_TRAIN_FLEX=1 "$PY" scripts/_hexfield_train_step_bench.py compiled shapes
echo '=== compiled(static-npad) + train_flex, big budget ==='
HEXFIELD_TRAIN_FLEX=1 "$PY" scripts/_hexfield_train_step_bench.py compiled big
