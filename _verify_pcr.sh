RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
for i in $(seq 1 25); do
  if [ "$(grep -cE 'RL start: RESUME' "$RD/rl_train.log" 2>/dev/null)" -ge 5 ]; then break; fi
  sleep 6
done
echo "=== startup: latest RESUME + config line ==="
grep -nE "RL start: RESUME|visits=[0-9]+ active=|eps=|dirichlet|PCR|pcr" "$RD/rl_train.log" 2>/dev/null | tail -4
echo "=== DRIVER ARGS (definitive — all three) ==="
ps -eo cmd | grep [_]rl_train_hexgnn | grep -oE "\-\-eps [0-9.]+|\-\-temperature-halflife [0-9.]+|\-\-temperature-floor [0-9.]+|\-\-pcr\b|\-\-pcr-full-proportion [0-9.]+|\-\-pcr-fast-visits [0-9]+|\-\-visits [0-9]+|\-\-n [0-9]+|\-\-widening-max-children [0-9]+"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
