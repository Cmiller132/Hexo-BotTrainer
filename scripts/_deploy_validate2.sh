#!/usr/bin/env bash
# Deploy validation round 2: new .so + copy-stream/train-compile sync.
set -eu
cp /tmp/hexshim/hexfield/_rust.cpython-312-x86_64-linux-gnu.so \
   /mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python/hexfield/_rust.cpython-312-x86_64-linux-gnu.so
cd /mnt/e/Hexo-BotTrainer-gumbel
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python

echo '--- import check (worktree) ---'
"$PY" -c "import hexfield.model, hexfield.inference, hexfield.trainer, hexfield._rust; print('imports OK')"
echo '--- save worktree flags-off baseline ---'
"$PY" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_hexfield_serve_ref.py save /tmp/main6_wt2_ref.npz
echo '--- worktree FULL STACK + copy_stream vs flags-off (5e-3) ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1 HEXFIELD_COPY_STREAM=1 \
  "$PY" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_hexfield_serve_ref.py check /tmp/main6_wt2_ref.npz 5e-3
