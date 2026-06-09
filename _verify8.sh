#!/bin/bash
sleep 14
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== seed source (should be epoch-8) + optimizer ==="
head -3 $R/rl_train.log 2>/dev/null
echo "=== supervisor.log ==="; head -1 $R/supervisor.log 2>/dev/null
echo "=== procs (driver + bridge) + RAM ==="
ps -eo pid,etimes,args | grep -E "_rl_train.py|_dashboard_bridge" | grep -v grep | cut -c1-90
free -g | grep Mem
