RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
sleep 360
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
echo "=== errors? ==="
grep -iE "Traceback|Error|Exception|FAILED|require_sealbot|SealBot.*fail|OOM|CUDA error" "$LOG" 2>/dev/null | grep -viE "no error|0 error|stderr=" | head -6 || echo "(none)"
echo "=== startup: from-scratch init / params / arch / visits / calibration ==="
grep -iE "scratch|random|initiali|no checkpoint|param|channel|block|search_visits|visits=|calibrat|batch|positions/sec|pos/s|target" "$LOG" 2>/dev/null | head -25
echo "=== self-play epoch 0 progress (positions / pos/s / shards) ==="
grep -iE "epoch 0|self.?play|selfplay|games|positions|pos/s" "$LOG" 2>/dev/null | tail -10
echo "=== run dir ==="; ls -R "$RD" 2>/dev/null | grep -vE "^$" | head -25
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
echo "=== supervisor.log tail ==="; tail -3 "$RD/supervisor.log" 2>/dev/null | sed -E 's/\[[0-9T:Z-]+\] //'
