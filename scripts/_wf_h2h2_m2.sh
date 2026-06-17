#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
exec /root/.venvs/hexgt-build/bin/python _wf_h2h2_arena.py \
  /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4/checkpoints/epoch_000013.pt \
  /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/checkpoints/epoch_000010.pt \
  m4ck13 m3ck10 96 _wf_h2h2_m2.json cuda 512 50 888000 256 \
  > _wf_h2h2_m2.log 2>&1
