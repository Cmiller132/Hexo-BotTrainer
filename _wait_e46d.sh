RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 38); do
  if grep -qE "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 10
done
echo "=== epoch-46 PCR summary ==="
grep -E "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "shards: $(ls "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | wc -l)/512"
