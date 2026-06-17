#!/bin/bash
RD=/mnt/e/Hexo-BotTrainer/runs/hexfield_soak
for i in $(seq 1 24); do
  if [ -f "$RD/diagnostics/hexfield.training.epoch_000001.json" ]; then
    echo "=== epoch-1 training diagnostic ==="
    cat "$RD/diagnostics/hexfield.training.epoch_000001.json"
    exit 0
  fi
  sleep 10
done
echo "training diag not yet present after 4 min; events tail:"
tail -3 "$RD/diagnostics/events.jsonl" 2>/dev/null | cut -c1-200
