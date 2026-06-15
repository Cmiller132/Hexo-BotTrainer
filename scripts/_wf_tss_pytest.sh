#!/usr/bin/env bash
# Scratch: run the TSS/FPU toggle tests + neighboring restnet config tests
# in the authoritative WSL venv. CPU-only (live run owns the GPU).
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_restnet_tss_fpu_toggle.py \
  tests/test_dense_cnn_restnet_pcr_policy_init.py \
  -q 2>&1 | tail -25
