#!/usr/bin/env bash
WD=165586
echo "=== count over 6s (stable single instance?) ==="
for i in 1 2 3; do
  echo "t$i: count=$(pgrep -fc _dashboard_bridge.py) pids=$(pgrep -f _dashboard_bridge.py | tr '\n' ' ')"
  sleep 2
done
echo "=== the surviving bridge ==="
for b in $(pgrep -f _dashboard_bridge.py); do
  ps -o pid,ppid,etimes,cmd --no-headers -p "$b"
done
echo "watchdog $WD alive: $(kill -0 $WD 2>/dev/null && echo yes || echo no)"
echo "=== driver (must be untouched) ==="
ps -o pid,etimes,cmd --no-headers -p "$(pgrep -f _rl_train.py | head -1)" | cut -c1-70
echo "=== mirror freshness (newest diagnostics file) ==="
ls -lat --time-style=+%H:%M:%S /mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main2/diagnostics/*.json | head -1
echo "=== bridge.log tail ==="
tail -2 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/bridge.log
