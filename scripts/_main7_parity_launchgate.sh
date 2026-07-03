#!/usr/bin/env bash
# LAUNCH GATE: serve parity of the full main_7 stack with the REAL prefit
# checkpoint at the main_7 arch, on the frozen main_6 state battery.
# Baseline = fp32 serve (triton conv + flex_pair, no half); check = launch
# stack (+ TRITON_ATTN + TRITON_CONV_LN + SERVE_HALF) at the 3e-3 gate.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
CKPT=${1:-/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7_prefit/checkpoint_epoch3.pt}

export HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=$CKPT
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1

echo '--- baseline save: fp32 serve, prefit ckpt, main_7 arch ---'
"$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_launch_base.npz
echo '--- LAUNCH GATE: full stack (attn+conv_ln+half) tol 3e-3 ---'
HEXFIELD_TRITON_ATTN=1 HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_SERVE_HALF=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_launch_base.npz 3e-3
echo LAUNCH GATE DONE
