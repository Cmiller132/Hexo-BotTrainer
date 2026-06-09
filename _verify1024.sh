RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 25); do
  if grep -qE "RL start: RESUME" "$RD/rl_train.log" 2>/dev/null && [ "$(grep -cE 'RL start' "$RD/rl_train.log")" -ge 3 ]; then break; fi
  sleep 6
done
echo "=== startup: RESUME + visits=1024 in log ==="
grep -nE "RL start: RESUME|visits=[0-9]+ active=" "$RD/rl_train.log" 2>/dev/null | tail -4
echo "=== driver process args (visits) ==="
ps -eo pid,cmd | grep _rl_train_hexgnn | grep -v grep | grep -oE "\-\-visits [0-9]+|\-\-eval-visits [0-9]+|\-\-active [0-9]+" | head
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
