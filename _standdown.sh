#!/bin/bash
# Stop supervisor (relauncher) FIRST, then driver, then bridge. Preserve all data.
# Supervisor bash reads the piped _rl_supervise.sh -> kill via the wrapper subtree + pattern.
for p in $(ps -eo pid,args | grep -iE "rl_supervise|_rl_run_fg" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
kill -9 2179407 2>/dev/null   # supervisor pid (best-effort)
sleep 1
pkill -TERM -f "_rl_train.py" 2>/dev/null
sleep 3
pkill -9 -f "_rl_train.py" 2>/dev/null
pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
sleep 2
echo "=== hexgt RL procs still running (want 0) ==="
ps -eo pid,args | grep -E "_rl_train.py|_rl_supervise|_dashboard_bridge|_rl_run_fg" | grep -v grep
echo "(blank above = none running)"
