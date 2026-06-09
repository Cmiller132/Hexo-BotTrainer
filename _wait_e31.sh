RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 40); do
  if grep -qE "epoch 31 selfplay:" "$RD/rl_train.log" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== DEFINITIVE n=5 + 1024-visit epoch-31 self-play ==="
grep -E "epoch 31 selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
grep -E "GPU mem \[epoch 31 selfplay\]" "$RD/rl_train.log" 2>/dev/null | tail -1 | sed -E 's/\[[0-9: -]+\] //'
echo "--- for comparison ---"
grep -E "epoch (29|30) selfplay:" "$RD/rl_train.log" 2>/dev/null | tail -2 | sed -E 's/\[[0-9: -]+\] //' | sed 's/ |.*cand=/ cand=/'
echo "=== health ==="
ps -eo etime,cmd | grep "[_]rl_train_hexgnn" | grep -v grep >/dev/null && echo "driver ALIVE" || echo "driver GONE"
grep -cE "RELAUNCH|HALT|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | sed 's/^/relaunch\/halt\/exc: /'
