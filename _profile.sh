RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
echo "=== sample GPU 4x over ~45s + log growth (CPU-prep vs GPU-train, progressing?) ==="
for i in 1 2 3 4; do
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)
  last=$(tail -1 "$LOG" 2>/dev/null | cut -c1-95)
  echo "[$i] GPU $gpu | log: $last"
  sleep 12
done
echo "=== what stage are the recent log lines (shuffle/data-prep vs train-step)? ==="
tail -20 "$LOG" 2>/dev/null | grep -ioE "compact_io|D6 coverage|shuffle|train|step|loss|epoch [0-9]+|policy|value" | sort | uniq -c | head
echo "=== epoch 5 training status now? (loss appeared?) ==="
/root/.venvs/hexgt-build/bin/python -c "import json;d=json.load(open('$RD/diagnostics/epoch_000005.json'));t=d['metadata']['result'].get('training',{});print('train.status=',t.get('status'),'loss=',t.get('loss'))" 2>/dev/null || echo "epoch_000005 result not finalized yet (train in progress)"
echo "=== sealbot worker still around? (eval vs train) ==="
ps -eo etime,cmd | grep -iE "_sealbot_worker" | grep -v grep | head -1 | cut -c1-60 || echo "(no sealbot worker -> not in eval)"
