#!/usr/bin/env bash
# Worker-like env: CPU only, radius per live runs, NO arch env (that is the
# point — the loader must infer the arch off the weights).
set -eu
export CUDA_VISIBLE_DEVICES=""
export HEXFIELD_SUPPORT_RADIUS=4
/root/.venvs/hexgt-build/bin/python /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_debug_screen_test.py
