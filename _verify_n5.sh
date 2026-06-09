RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 25); do
  if [ "$(grep -cE 'RL start: RESUME' "$RD/rl_train.log" 2>/dev/null)" -ge 3 ]; then break; fi
  sleep 6
done
echo "=== startup log: latest RESUME + n + visits ==="
grep -nE "RL start: RESUME|visits=[0-9]+ active=" "$RD/rl_train.log" 2>/dev/null | tail -2
echo "=== driver process args (n / widening / visits / eval-visits) ==="
ps -eo cmd | grep _rl_train_hexgnn | grep -v grep | grep -oE "\-\-n [0-9]+|\-\-widening-max-children [0-9]+|\-\-visits [0-9]+|\-\-eval-visits [0-9]+|\-\-active [0-9]+" | head
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
