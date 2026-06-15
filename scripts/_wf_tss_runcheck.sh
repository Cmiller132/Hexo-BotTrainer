#!/usr/bin/env bash
# Scratch: is the main_4 driver alive and what phase is it in?
set -u
cd /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
pid=$(tr -d '[:space:]' < driver.pid 2>/dev/null)
echo "driver.pid=$pid"
if ps -p "$pid" > /dev/null 2>&1; then
  echo "driver ALIVE: $(ps -p "$pid" -o etime= | tr -d ' ')"
else
  echo "driver DEAD"
fi
echo "supervisor flags:"
ls supervisor* 2>/dev/null
echo "halt flag: $(ls supervisor_halted.flag 2>/dev/null || echo none)"
echo "train procs: $(ps aux | grep -c '[t]rain_model')"
echo "latest run-dir writes:"
ls -lt checkpoints/ | head -3
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
