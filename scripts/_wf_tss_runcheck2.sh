#!/usr/bin/env bash
# Scratch: full main_4 liveness forensic -> file (avoids wsl.exe output mangling).
set -u
out=/mnt/e/Hexo-BotTrainer-hexgt/scripts/_wf_tss_runcheck2.txt
cd /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
{
  echo "=== $(date -u) ==="
  pid=$(tr -d '[:space:]' < driver.pid 2>/dev/null || echo none)
  echo "driver.pid file: $pid"
  if [ "$pid" != "none" ] && ps -p "$pid" > /dev/null 2>&1; then
    echo "driver process: ALIVE ($(ps -p "$pid" -o etime=,cmd= 2>/dev/null))"
  else
    echo "driver process: DEAD"
  fi
  echo "--- run-dir files ---"
  ls -la *.lock *.pid *.flag supervisor.log 2>/dev/null
  echo "--- supervisor.lock content ---"
  cat supervisor.lock 2>/dev/null || echo "(no supervisor.lock)"
  spid=$(head -1 supervisor.lock 2>/dev/null | tr -d '[:space:]')
  if [ -n "${spid:-}" ] && ps -p "$spid" > /dev/null 2>&1; then
    echo "supervisor process: ALIVE"
  else
    echo "supervisor process: DEAD or unknown"
  fi
  echo "--- supervisor.log tail ---"
  tail -30 supervisor.log 2>/dev/null
  echo "--- driver log tail (newest driver_*.log / train log) ---"
  newest=$(ls -t *.log logs/*.log 2>/dev/null | head -2)
  for f in $newest; do echo "## $f"; tail -40 "$f"; done
  echo "--- any python/train processes ---"
  ps aux | grep -E "[t]rain_model|[h]exo_train" || echo "(none)"
} > "$out" 2>&1
echo "written"
