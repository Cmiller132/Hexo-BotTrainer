RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 220); do
  if grep -qE "epoch 30 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== epoch 30 selfplay (visits=1024) vs prior (512) ==="
grep -E "epoch (28|29|30) selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -3
grep -E "GPU mem \[epoch 30 selfplay\]" "$RD/rl_train.log" 2>/dev/null | tail -1
