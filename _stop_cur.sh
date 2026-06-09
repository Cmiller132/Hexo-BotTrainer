#!/bin/bash
pkill -TERM -f "_rl_train.py" 2>/dev/null; sleep 3
kill -9 2172879 2172908 2172986 2>/dev/null
pkill -9 -f "_rl_train.py" 2>/dev/null; pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
for p in $(ps -eo pid,args | grep -iE "rl_supervise|_rl_run_fg" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
sleep 2
echo "survivors: $(ps -eo pid,args | grep -E '_rl_train.py|_dashboard_bridge|rl_supervise' | grep -v grep | wc -l)"
