#!/usr/bin/env bash
# Detect the 96x8 epoch training phase (expansion pool spawns), sample GPU util,
# and time the phase end-to-end (pool drains) to confirm the parallel speedup.
set -uo pipefail
nw=0
tp=""
for i in $(seq 1 600); do
  tp=$(pgrep -f 'train_model.*96x8' | head -1)
  nw=0
  if [ -n "$tp" ]; then nw=$(pgrep -P "$tp" 2>/dev/null | wc -l); fi
  if [ "${nw:-0}" -gt 0 ]; then break; fi
  sleep 10
done
if [ "${nw:-0}" -le 0 ]; then echo "POOL_NEVER_SPAWNED"; exit 1; fi
start=$(date +%s)
echo "TRAIN_START $(date +%T) trainer=${tp} workers=${nw}"
for s in $(seq 1 6); do
  echo "gpu $(date +%T): $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
  sleep 4
done
for i in $(seq 1 300); do
  nw=$(pgrep -P "$tp" 2>/dev/null | wc -l)
  if [ "${nw:-0}" -eq 0 ]; then break; fi
  sleep 3
done
end=$(date +%s)
echo "TRAIN_END $(date +%T) training_phase_duration_s=$((end - start))"
