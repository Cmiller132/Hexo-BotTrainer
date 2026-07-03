#!/usr/bin/env bash
# Isolate the step-B diff: (1) dev save + dev check must be ~0 (determinism);
# (2) HEAD model.py (pre-trunk-rewrite) vs working-copy model.py, same
# everything else -> attributes the diff to the loop rewrite or not.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1

echo '--- 1: dev save ---'
"$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_iso_dev.npz
echo '--- 1b: dev self-check (determinism, tol 1e-6) ---'
"$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 1e-6

echo '--- 2: build HEAD-model.py package copy ---'
rm -rf /tmp/hexfield_head_pkg
mkdir -p /tmp/hexfield_head_pkg
cp -r packages/hexfield/python/hexfield /tmp/hexfield_head_pkg/
git show HEAD:packages/hexfield/python/hexfield/model.py \
  > /tmp/hexfield_head_pkg/hexfield/model.py
git show HEAD:packages/hexfield/python/hexfield/constants.py \
  > /tmp/hexfield_head_pkg/hexfield/constants.py
echo '--- 2b: HEAD model.py vs dev baseline (tol 1e-6) ---'
HEXFIELD_REF_PKGROOT=/tmp/hexfield_head_pkg \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 1e-6
echo ISO DONE
