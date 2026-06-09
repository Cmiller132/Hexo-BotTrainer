#!/bin/bash
# Kill the RL run tree: supervisor FIRST (so it cannot relaunch), then driver(s),
# then the wrapper. Literal pids (avoids the WSL inline-$var stripping gotcha).
kill -9 2110060 2>/dev/null   # supervisor (relauncher)
sleep 1
pkill -9 -f _rl_train.py 2>/dev/null
kill -9 2110048 2>/dev/null   # wrapper
sleep 2
echo "=== survivors ==="
ps -eo pid,args 2>/dev/null | grep -E "_rl_train.py|_rl_supervise|_rl_run_fg.sh" | grep -v grep | head
echo "(none above = clean)"
