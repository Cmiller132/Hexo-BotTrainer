RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 180); do
  if grep -qE "epoch 3 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 10
done
echo "=== epoch 3 selfplay (NEW build, uncontended) ==="
grep -E "epoch [0-9]+ selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -4
grep -E "GPU mem \[epoch 3 selfplay\]" "$RD/rl_train.log" 2>/dev/null | tail -1
