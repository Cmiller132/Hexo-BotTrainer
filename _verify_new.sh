#!/bin/bash
sleep 12
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== new run config header ==="
cat $R/rl_train.log 2>/dev/null | head -2
echo "=== supervisor.log ==="
cat $R/supervisor.log 2>/dev/null | head -3
echo "=== procs: supervisor + driver + bridge ==="
ps -eo pid,etimes,args | grep -E "_rl_supervise|_rl_train.py|_dashboard_bridge" | grep -v grep | cut -c1-95
echo "=== bridge pointed at main2? ==="
grep -c "main2" /mnt/e/Hexo-BotTrainer-hexgt/_dashboard_bridge.py
echo "=== RAM ==="
free -g | grep Mem
