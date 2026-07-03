#!/usr/bin/env bash
# Parity gates for HEXFIELD_TRITON_CONV_LN vs the dev baselines saved earlier
# (/tmp/m7_iso_dev.npz fp32-serve, /tmp/m7_iso_half.npz serve-half).
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1

echo '--- E: + TRITON_CONV_LN (tol 3e-3) ---'
HEXFIELD_TRITON_CONV_LN=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 3e-3
echo '--- F: full main_7 serve stack: CONV_LN + TRITON_ATTN + SERVE_HALF (tol 3e-3) ---'
HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_TRITON_ATTN=1 HEXFIELD_SERVE_HALF=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_half.npz 3e-3
echo CONVLN GATES DONE
