#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
exec /root/.venvs/hexgt-build/bin/python _wf_h2h2_arena.py \
  /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4/checkpoints/epoch_000013.pt \
  /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/checkpoints/epoch_000005.pt \
  m4ck13 m3ck5 96 _wf_h2h2_m1.json cuda 512 55 777000 256 \
  > _wf_h2h2_m1.log 2>&1
