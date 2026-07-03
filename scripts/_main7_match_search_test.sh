#!/usr/bin/env bash
set -eu
export CUDA_VISIBLE_DEVICES=""
export HEXFIELD_SUPPORT_RADIUS=4
/root/.venvs/hexgt-build/bin/python /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_match_search_test.py
