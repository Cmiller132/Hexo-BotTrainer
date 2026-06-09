#!/usr/bin/env bash
# Poll the current main3 driver log for self-play pos/s throughput.
set -uo pipefail
LOG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3/driver.20260604_023004.out.log
DRV=54299
for i in $(seq 1 18); do
  sleep 20
  if grep -qiE 'pos/s|positions/s' "$LOG" 2>/dev/null; then break; fi
  if ! kill -0 "$DRV" 2>/dev/null; then echo "DRIVER GONE"; break; fi
done
echo "=== driver tail ==="
tail -14 "$LOG"
echo "=== pos/s lines ==="
grep -iE 'pos/s|positions/s' "$LOG" | tail -6
echo "=== alive? ==="
kill -0 "$DRV" 2>/dev/null && echo "ALIVE pid=$DRV" || echo "GONE"
