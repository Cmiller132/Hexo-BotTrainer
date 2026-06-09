RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 46); do
  if grep -qE "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== DEFINITIVE epoch-46 PCR self-play summary (pos/s + pcr_str) ==="
grep -E "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
grep -E "GPU mem \[epoch 46 selfplay\]" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "--- prior full-1024 (n=4) epochs for comparison ---"
grep -E "epoch (44|45) selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -2 | sed -E 's/\[[0-9: -]+\] //' | sed -E 's/ \| Q3.*//'
echo "=== health ==="
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
