#!/usr/bin/env bash
# Wait for the batch-32 prefit to finish (VERIFIED / checkpoint) or fail (OOM/error),
# sampling VRAM every ~60s. Reports the trajectory + final log tail.
set -uo pipefail
PRE=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit
LOG="$PRE/prefit.log"
PID="${1:-224858}"
start=$(date +%s)
for i in $(seq 1 330); do
  if grep -qiE 'VERIFIED|wrote checkpoint|Traceback|device not ready|out of memory|OutOfMemory|CUDA error|FATAL' "$LOG" 2>/dev/null; then echo "DONE: terminal-marker"; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "DONE: process-gone"; break; fi
  if (( i % 6 == 0 )); then
    now=$(date +%s); el=$((now-start))
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    ep=$(grep -cE 'prefit\] epoch' "$LOG" 2>/dev/null || echo 0)
    echo "[+${el}s] GPU: $gpu | epoch-lines=$ep"
  fi
  sleep 10
done
echo "=== prefit.log tail ==="
tail -14 "$LOG"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
end=$(date +%s); echo "watch elapsed: $((end-start))s"
