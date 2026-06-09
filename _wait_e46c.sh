RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 40); do
  if grep -qE "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 10
done
echo "=== DEFINITIVE epoch-46 PCR self-play summary ==="
grep -E "epoch 46 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "(prior full-1024 n=4: epoch 45 = 18.0 pos/s)"
ps -eo cmd | grep -q "[_]rl_train_hexgnn" && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
