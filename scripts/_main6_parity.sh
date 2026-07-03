#!/usr/bin/env bash
# Parity gate for the flex-pair serve path against real main_6 states + ckpt.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python

echo '--- save baseline (flex_pair OFF) ---'
"$PY" scripts/_hexfield_serve_ref.py save /tmp/main6_ref.npz
echo '--- check flex_pair ON (tol 3e-3) ---'
HEXFIELD_FLEX_PAIR=1 "$PY" scripts/_hexfield_serve_ref.py check /tmp/main6_ref.npz 3e-3
echo '--- self-check flags OFF (must be ~0) ---'
"$PY" scripts/_hexfield_serve_ref.py check /tmp/main6_ref.npz 1e-6
