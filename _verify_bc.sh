#!/bin/bash
sleep 14
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== seed source (should be hexgt_bc_step006009, NOT epoch-8) ==="
head -2 $R/rl_train.log 2>/dev/null
echo "=== procs ==="
ps -eo pid,etimes,args | grep -E "_rl_supervise|_rl_train.py|_dashboard_bridge" | grep -v grep | cut -c1-90
echo "=== RAM ==="; free -g | grep Mem
