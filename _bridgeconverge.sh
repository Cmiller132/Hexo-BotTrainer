#!/usr/bin/env bash
SUPLOG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/supervisor.log
WD=$(grep -oE "bridge watchdog started \(pid=[0-9]+" "$SUPLOG" | tail -1 | grep -oE "[0-9]+$")
echo "latest watchdog pid (parent of the durable bridge) = $WD"
echo "=== bridges now ==="
for b in $(pgrep -f _dashboard_bridge.py); do
  ppid=$(ps -o ppid= -p "$b" | tr -d ' ')
  start=$(ps -o lstart= -p "$b")
  if [ "$ppid" = "$WD" ]; then tag="KEEP (watchdog child)"; else tag="ad-hoc -> will kill"; fi
  echo "bridge $b  ppid=$ppid  start=$start  -> $tag"
done
