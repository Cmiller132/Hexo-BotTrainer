#!/usr/bin/env bash
# Kernel gates vs the dev-pkg compiled baseline (/tmp/m7_iso_dev.npz from
# _main7_parity_isolate.sh step 1).
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1

echo '--- C: + TRITON_ATTN vs flex baseline (tol 3e-3) ---'
HEXFIELD_TRITON_ATTN=1 "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 3e-3
echo '--- D1: SERVE_HALF baseline save ---'
HEXFIELD_SERVE_HALF=1 "$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_iso_half.npz
echo '--- D2: SERVE_HALF + TRITON_ATTN (tol 3e-3) ---'
HEXFIELD_SERVE_HALF=1 HEXFIELD_TRITON_ATTN=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_half.npz 3e-3
echo GATES DONE
