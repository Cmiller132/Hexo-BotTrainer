#!/bin/bash
# Stop supervisor first (no relaunch), then driver; stop the bridge too.
kill -TERM 2160571 2>/dev/null; sleep 2
kill -TERM 2160600 2>/dev/null; sleep 3
kill -9 2160571 2160600 2>/dev/null
pkill -9 -f "_rl_train.py" 2>/dev/null
pkill -9 -f "_rl_supervise.sh" 2>/dev/null
pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
sleep 2
echo "=== survivors (none=clean) ==="
ps -eo pid,args | grep -E "_rl_train.py|_rl_supervise|_dashboard_bridge" | grep -v grep | head
