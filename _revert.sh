#!/bin/bash
# Stop the epoch-8-seeded main2 (supervisor first), the bridge, then wipe + relaunch BC-seeded.
kill -TERM 2164762 2>/dev/null; sleep 2
kill -TERM 2164791 2>/dev/null; sleep 3
kill -9 2164762 2164791 2>/dev/null
pkill -9 -f "_rl_train.py" 2>/dev/null; pkill -9 -f "_rl_supervise.sh" 2>/dev/null; pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
sleep 2
echo "survivors (none=clean): $(ps -eo pid,args | grep -E '_rl_train.py|_rl_supervise|_dashboard_bridge' | grep -v grep | wc -l)"
