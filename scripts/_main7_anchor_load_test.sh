#!/usr/bin/env bash
# Run with the LIVE main_7 trainer's arch env — the situation the epoch-5
# multistage eval will be in when it loads foreign anchors.
set -eu
export CUDA_VISIBLE_DEVICES=""
export HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA
/root/.venvs/hexgt-build/bin/python /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_anchor_load_test.py
