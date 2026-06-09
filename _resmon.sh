#!/usr/bin/env bash
# Lightweight resource monitor for the hexgt RL run: every 60s append a JSON line
# with free system RAM + GPU memory used + GPU util. Read-only; no GPU work.
set -uo pipefail
OUT=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/watch_resources.jsonl
mkdir -p "$(dirname "$OUT")"
while :; do
  ts=$(date -u +%FT%TZ)
  free_ram=$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)
  tot_ram=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
  read gutil gmem <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')"
  echo "{\"t\":\"$ts\",\"free_ram_gb\":$free_ram,\"tot_ram_gb\":$tot_ram,\"gpu_util\":${gutil:-0},\"gpu_mem_used_mb\":${gmem:-0}}" >> "$OUT"
  sleep 60
done
