#!/usr/bin/env bash
# Eager-mode (HEXFIELD_NO_COMPILE=1) comparison: HEAD model.py vs working-copy
# model.py. Identical -> the compiled-mode 6e-4 is inductor codegen variance,
# not a math change from the trunk rewrite.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_NO_COMPILE=1

echo '--- eager save (HEAD pkg) ---'
HEXFIELD_REF_PKGROOT=/tmp/hexfield_head_pkg \
  "$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_iso_eager.npz
echo '--- eager check (dev pkg, tol 1e-6) ---'
"$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_eager.npz 1e-6
echo ISO2 DONE
