bash /tmp/_hexgnn_env.sh /mnt/e/Hexo-BotTrainer-hexgt/scripts/_rl_train_hexgnn.py \
  --device cpu --no-compile \
  --bc-seed /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke/pretrain/seed.pt \
  --out-dir /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_smoke \
  --epochs 1 --games-per-epoch 2 --active 2 --vbatch 2 --visits 6 --max-actions 30 \
  --train-steps-per-epoch 3 --batch 8 --eval-games 0 --no-policy-surprise --n 3
