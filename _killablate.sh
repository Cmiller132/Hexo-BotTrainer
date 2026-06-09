#!/bin/bash
kill -9 2120890 2120882 2120876 2>/dev/null
sleep 1
pkill -9 -f _rl_ablate.py 2>/dev/null
pkill -9 -f _ablate_run.sh 2>/dev/null
sleep 2
echo "=== survivors ==="
ps -eo pid,args 2>/dev/null | grep -E "_rl_ablate|_ablate_run" | grep -v grep | head
echo "(none above = clean)"
