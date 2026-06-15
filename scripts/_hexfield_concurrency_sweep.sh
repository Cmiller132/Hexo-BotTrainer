#!/usr/bin/env bash
# Throughput vs concurrency sweep: run the real continuous scheduler at
# GAMES in {16,32,64,128} with production search params (512v, PCR128@33%),
# warm-started from the BC ckpt. Reveals whether evals/s scales with the
# active-game pool or collapses (memory-pressure thrash) at 128.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export OMP_NUM_THREADS=8
CKPT=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1/checkpoint_epoch2.pt
PY=/root/.venvs/hexgt-build/bin/python
for G in 16 32 64 128; do
  echo "############### GAMES=$G ###############"
  PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 "$G" 24 2>&1 \
    | grep -E 'games=|DECISIONS/S|EVALS/S|mean_flush|early_stops|peak VRAM'
  echo
done
