#!/bin/bash
sleep 14
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== run header (seed source + config) ==="
head -4 $R/rl_train.log 2>/dev/null
echo "=== supervisor.log ==="
head -2 $R/supervisor.log 2>/dev/null
echo "=== procs: supervisor + driver + bridge ==="
ps -eo pid,etimes,args | grep -E "_rl_supervise|_rl_train.py|_dashboard_bridge" | grep -v grep | cut -c1-95
echo "=== bridge pointed at main2? + RAM ==="
grep -c "rl_main2" /mnt/e/Hexo-BotTrainer-hexgt/_dashboard_bridge.py
free -g | grep Mem
