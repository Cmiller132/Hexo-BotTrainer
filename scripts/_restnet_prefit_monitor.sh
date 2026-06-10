#!/usr/bin/env bash
# Wait until the prefit logs its first training epoch (or errors/exits), then
# report the log tail + GPU. Arg1: target marker regex (default first epoch).
set -uo pipefail
PRE=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit
LOG="$PRE/prefit.log"
PID="${2:-224041}"
TARGET="${1:-prefit\] epoch 0:}"
for i in $(seq 1 600); do
  if grep -qE "$TARGET" "$LOG" 2>/dev/null; then echo "HIT: target ($TARGET)"; break; fi
  if grep -qiE 'Traceback|out of memory|OutOfMemory|CUDA error|FATAL|VERIFIED|wrote checkpoint' "$LOG" 2>/dev/null; then echo "HIT: terminal-marker"; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "HIT: process-gone (pid $PID)"; break; fi
  sleep 5
done
echo "=== prefit.log tail ==="
tail -20 "$LOG"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo "=== process ==="
kill -0 "$PID" 2>/dev/null && echo "ALIVE pid=$PID" || echo "GONE pid=$PID"
