#!/usr/bin/env bash
# Measure fp8 conv deviation vs the dev baselines. fp8 routes ONLY through the
# trunk ConvBlocks' fused conv+LN path (head convs stay fp16 — measured: fp8
# heads tripled prior deviation). NOT a 3e-3 gate — the real gate is the arena
# eval at main_7 bring-up; 5e-2 here is a broken-kernel sanity ceiling.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1

echo '--- G1: CONV_LN + FP8 (trunk-only fp8) vs fp32 baseline (5e-2) ---'
HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_CONV_FP8=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 5e-2
echo '--- G2: full stack + fp8: CONV_LN + FP8 + TRITON_ATTN + SERVE_HALF (5e-2) ---'
HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_CONV_FP8=1 HEXFIELD_TRITON_ATTN=1 HEXFIELD_SERVE_HALF=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_half.npz 5e-2
echo FP8 MEASURE DONE
