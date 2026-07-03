#!/usr/bin/env bash
# Validate the deployed gumbel-worktree serve stack: import + parity vs the
# flags-off baseline (/tmp/main6_ref.npz), running FROM the worktree.
set -eu
cd /mnt/e/Hexo-BotTrainer-gumbel
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt
# hexfield from the WORKTREE (the code under test); engine/runner/utils from
# the dev tree (they carry the built native modules).
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python

echo '--- import check ---'
"$PY" -c "import hexfield.model, hexfield.inference, hexfield._triton_conv; print('imports OK')"
echo '--- save worktree flags-off baseline (frozen states) ---'
"$PY" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_hexfield_serve_ref.py save /tmp/main6_wt_ref.npz
echo '--- worktree FULL STACK vs flags-off ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1 \
  "$PY" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_hexfield_serve_ref.py check /tmp/main6_wt_ref.npz 5e-3
