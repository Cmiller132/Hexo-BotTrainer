GLOG=/mnt/e/Hexo-BotTrainer-hexgt/_dc_gate.out
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
sleep 300
echo "=== gate process alive? ==="
ps -eo pid,etime,cmd | grep "[h]exo_train.cli.train_model" | grep -oE "^ *[0-9]+ +[0-9:]+" | head -1 || echo "NOT RUNNING"
echo "=== ERRORS/tracebacks? ==="
grep -iE "Traceback|Error|Exception|FAILED|require_sealbot|SealBot.*(fail|launch)|CUDA|OOM" "$GLOG" 2>/dev/null | grep -vE "no error|stderr" | head -8 || echo "(none)"
echo "=== startup / init / calibration / self-play markers ==="
grep -iE "scratch|random|init|param|visits|calibrat|pos/s|positions|epoch 0|self.?play|selfplay|search_visits|channels|block" "$GLOG" 2>/dev/null | head -25
echo "=== run dir contents ==="
ls -R "$RD" 2>/dev/null | head -20
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
