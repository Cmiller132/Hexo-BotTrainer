#!/usr/bin/env bash
# Active poll for prefit completion: returns the moment the checkpoint exists /
# VERIFIED, else reports progress and keeps the caller polling. ~8 min window.
PRE=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit
LOG="$PRE/prefit.log"
PID="${1:-224858}"
for i in $(seq 1 9); do
  if [[ -f "$PRE/restnet_hf_prefit.pt" ]] || grep -qE 'VERIFIED' "$LOG" 2>/dev/null; then
    echo "=== PREFIT DONE ==="
    grep -E 'shuffled|prefit\] epoch|wrote checkpoint|VERIFIED' "$LOG"
    ls -la "$PRE/restnet_hf_prefit.pt" 2>/dev/null
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "=== PROCESS GONE (no checkpoint) ==="; tail -25 "$LOG"; exit 2
  fi
  last=$(grep -oE 'data000[0-9]+\.npz' "$LOG" 2>/dev/null | tail -1)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  echo "[poll $i] alive last_shard=$last GPU=$gpu"
  sleep 55
done
echo "=== STILL RUNNING (re-poll) ==="
grep -E 'shuffled|prefit\] epoch' "$LOG" | tail -2
