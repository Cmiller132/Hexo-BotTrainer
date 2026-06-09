#!/usr/bin/env bash
# Watch the prefit: sample VRAM periodically, exit when epoch 0 completes / errors /
# process dies. Reports the VRAM trajectory + log tail.
set -uo pipefail
PRE=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit
LOG="$PRE/prefit.log"
PID="${1:-224500}"
start=$(date +%s)
for i in $(seq 1 240); do
  if grep -qE 'prefit\] epoch 0:' "$LOG" 2>/dev/null; then echo "DONE: epoch0 line"; break; fi
  if grep -qiE 'Traceback|out of memory|device not ready|OutOfMemory|CUDA error|FATAL|VERIFIED' "$LOG" 2>/dev/null; then echo "DONE: terminal-marker"; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "DONE: process-gone"; break; fi
  if (( i % 6 == 0 )); then
    now=$(date +%s); el=$((now-start))
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    echo "[+${el}s] GPU: $gpu"
  fi
  sleep 5
done
echo "=== prefit.log tail ==="
tail -16 "$LOG"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
end=$(date +%s); echo "watch elapsed: $((end-start))s"
