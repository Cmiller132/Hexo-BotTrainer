RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 25); do
  if [ "$(grep -cE 'RL start: RESUME' "$RD/rl_train.log" 2>/dev/null)" -ge 4 ]; then break; fi
  sleep 6
done
echo "=== startup: latest RESUME + n + visits ==="
grep -nE "RL start: RESUME|visits=[0-9]+ active=" "$RD/rl_train.log" 2>/dev/null | tail -2
echo "=== driver args ==="
ps -eo cmd | grep [_]rl_train_hexgnn | grep -oE "\-\-n [0-9]+|\-\-widening-max-children [0-9]+|\-\-visits [0-9]+|\-\-eval-visits [0-9]+|\-\-active [0-9]+|\-\-soft-z-lambda [0-9]+" | head
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
