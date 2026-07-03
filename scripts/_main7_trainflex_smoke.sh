#!/usr/bin/env bash
set -eu
export HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1 HEXFIELD_TRAIN_FLEX=1
export HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA
/root/.venvs/hexgt-build/bin/python /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_trainflex_smoke.py
echo '--- serve parity unchanged (flags-identical self-check vs saved baseline):'
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_ATTENTION_HEADS=4 HEXFIELD_TRUNK=CCCACCCACCA
export HEXFIELD_REF_RUN=/tmp/main7_ref_run
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 \
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_serve_ref.py check /tmp/m7_iso_dev.npz 1e-6
