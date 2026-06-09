RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 20); do
  if grep -qE "RESUME from|FROM SCRATCH|hexgnn RL start" "$RD/rl_train.log" 2>/dev/null && [ "$(grep -cE 'hexgnn RL start' "$RD/rl_train.log")" -ge 2 ]; then break; fi
  sleep 6
done
echo "=== driver startup (latest 'RL start' block) ==="
grep -nE "RESUME from|FROM SCRATCH|hexgnn RL start|params=|RESUME LOAD|epoch 3 selfplay" "$RD/rl_train.log" 2>/dev/null | tail -6
echo "=== new processes ==="
ps -eo pid,pgid,etime,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep | head
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
