RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 46); do
  if grep -qE "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== DEFINITIVE epoch-46 PCR summary ==="
grep -E "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "--- comparison: full-1024 n=4 ---"
grep -E "epoch 45 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //' | sed -E 's/ \| Q3.*//'
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
