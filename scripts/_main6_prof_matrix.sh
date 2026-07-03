#!/usr/bin/env bash
# A/B matrix for the serve-forward levers (run in WSL, GPU free).
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SERVE_FLEX=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
PY=/root/.venvs/hexgt-build/bin/python

echo '=== half + triton_conv + flex_pair (compiled) ==='
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 "$PY" scripts/_hexfield_main6_profile.py compiled half
