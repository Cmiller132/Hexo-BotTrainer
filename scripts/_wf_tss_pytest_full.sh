#!/usr/bin/env bash
# Scratch: full restnet-family pytest after the TSS/FPU toggle plumbing.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_restnet*.py \
  tests/test_dense_cnn_continuous_scheduler.py \
  tests/test_restnet_reference_eval.py \
  -q 2>&1 | tail -8
