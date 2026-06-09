RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 25); do
  if [ "$(grep -cE 'RL start: RESUME' "$RD/rl_train.log" 2>/dev/null)" -ge 6 ]; then break; fi
  sleep 6
done
echo "=== startup: latest RESUME + config line ==="
grep -E "RL start: RESUME|visits=[0-9]+ active=" "$RD/rl_train.log" 2>/dev/null | tail -2 | sed -E 's/\[[0-9: -]+\] //'
echo "=== driver args (confirm NO --pcr + settings) ==="
echo "  --pcr present? $(ps -eo cmd | grep [_]rl_train_hexgnn | grep -v grep | grep -cE ' --pcr')"
ps -eo cmd | grep [_]rl_train_hexgnn | grep -v grep | grep -oE "\-\-n [0-9]+|\-\-visits [0-9]+|\-\-eps [0-9.]+|\-\-temperature-halflife [0-9.]+|\-\-temperature-floor [0-9.]+|\-\-widening-max-children [0-9]+|\-\-active [0-9]+"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
