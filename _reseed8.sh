#!/bin/bash
# Stop current BC-seeded main2 (supervisor first via pattern), driver, bridge.
pkill -TERM -f "_rl_train.py" 2>/dev/null
sleep 3
kill -9 2167867 2167942 2>/dev/null
pkill -9 -f "_rl_train.py" 2>/dev/null; pkill -9 -f "tr -d" 2>/dev/null; pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
# kill the supervisor bash loop (reads piped _rl_supervise.sh) by its child relationship — find any bash running supervise
for p in $(ps -eo pid,args | grep -i "rl_supervise\|_rl_run_fg" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 2
echo "survivors: $(ps -eo pid,args | grep -E '_rl_train.py|_dashboard_bridge' | grep -v grep | wc -l)"
