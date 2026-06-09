RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== supervisor.log tail (HALT/breaker/crashes/exit?) ==="
tail -12 "$RD/supervisor.log" 2>/dev/null | sed -E 's/\[[0-9T:Z-]+\] //'
echo "=== crash artifacts? ==="
ls -t "$RD/crash_artifacts"/ 2>/dev/null | head -3 || echo "(none)"
echo "=== latest train.out.log tail (driver exit reason?) ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1); echo "log: $(basename $LOG)"
tail -8 "$LOG" 2>/dev/null | grep -vE "frombuffer|writable|tensor_new|compact_io" | cut -c1-120
echo "=== was there a HALT flag before I wrote one? (supervisor.log would show breaker) ==="
grep -E "HALT|breaker tripped|EXIT.*code=[^0]|EXCEPTION" "$RD/supervisor.log" 2>/dev/null | tail -4 | sed -E 's/\[[0-9T:Z-]+\] //'
echo "=== epoch 7 status (did it crash mid-epoch?) ==="
ls -t "$RD/diagnostics"/dense_cnn.evaluation.epoch_000007.json "$RD/diagnostics"/epoch_000007.json 2>/dev/null | xargs -n1 basename 2>/dev/null || echo "no epoch_7 result/eval yet (died after selfplay, before train/eval done)"
