#!/usr/bin/env bash
set -e
cd /mnt/e/Hexo-BotTrainer-hexgt
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1/checkpoints/epoch_000033.pt
TAG="$1"   # base or fix
for r in 1 2 3; do
  HEXFIELD_ASYNC_EVAL=1 PYTHONPATH=packages/hexfield/python \
    /root/.venvs/hexgt-build/bin/python scripts/_hexfield_lategame_bench.py \
    "$CKPT" 75 "96" "/tmp/${TAG}_${r}.json" 2>&1 | grep -E "pos/s="
done
