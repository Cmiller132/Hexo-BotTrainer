RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "now: $(date '+%H:%M:%S')"
echo "=== train.out.log tail (actual stage activity) ==="
tail -12 "$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)" 2>/dev/null | grep -vE "frombuffer|writable|tensor_new" | cut -c1-160
echo "=== SealBot / eval subprocesses running? ==="
ps -eo pid,etime,cmd 2>/dev/null | grep -iE "sealbot|seal_bot" | grep -v grep | head -3 || echo "(no sealbot proc)"
echo "=== driver CPU/state (is it working?) ==="
ps -eo pid,etime,%cpu,stat,cmd 2>/dev/null | grep [h]exo_train.cli.train_model | cut -c1-70
echo "=== most-recently-modified run-dir files (is it writing?) ==="
ls -lt --time-style=+%H:%M:%S "$RD"/diagnostics/* "$RD"/checkpoints/* "$RD"/shuffleddata/* 2>/dev/null | head -6
echo "=== shuffleddata / samples dirs ==="
du -sh "$RD/shuffleddata" "$RD/samples" 2>/dev/null
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)"
