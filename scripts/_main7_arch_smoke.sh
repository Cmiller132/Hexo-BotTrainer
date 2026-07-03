#!/usr/bin/env bash
# Smoke: layout-driven trunk. main_6 compat (ckpt strict load) + main_7 build.
set -eu
PY=/root/.venvs/hexgt-build/bin/python
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
export CUDA_VISIBLE_DEVICES=""
cd /mnt/e/Hexo-BotTrainer-hexgt

HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_CHANNELS=128 \
  "$PY" scripts/_main7_arch_smoke.py main6 \
  /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
echo =====
HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 \
  HEXFIELD_TRUNK=CCACCACCACCACCA \
  "$PY" scripts/_main7_arch_smoke.py main7
