#!/bin/bash
LOG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/bc_train.log
for i in $(seq 1 180); do
  if grep -q "=== DONE" "$LOG" 2>/dev/null; then echo "BC_FINISHED"; grep "HELDOUT" "$LOG" | tail -8; exit 0; fi
  n=$(grep -c "HELDOUT" "$LOG" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 5 ]; then echo "BC_5_EVALS"; grep "HELDOUT" "$LOG" | tail -6; exit 0; fi
  sleep 60
done
echo "WAIT_TIMEOUT"; grep "HELDOUT" "$LOG" | tail -8
