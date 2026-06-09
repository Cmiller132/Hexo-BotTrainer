#!/bin/bash
sleep 14
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== header (seed + visits=512?) ==="
head -2 $R/rl_train.log 2>/dev/null
echo "=== supervisor.log ==="; head -1 $R/supervisor.log 2>/dev/null
echo "=== procs + RAM ==="
ps -eo pid,etimes,args | grep -E "_rl_supervise|_rl_train.py|_dashboard_bridge" | grep -v grep | cut -c1-92
free -g | grep Mem
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
