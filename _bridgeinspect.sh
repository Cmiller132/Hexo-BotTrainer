#!/usr/bin/env bash
echo "=== all _dashboard_bridge.py processes ==="
for b in $(pgrep -f _dashboard_bridge.py); do
  ps -o pid,ppid,etimes,cmd --no-headers -p "$b"
done
echo "=== parent identification ==="
echo "my supervisor pid (from log): grep below"
grep -E "SUPERVISOR start|bridge watchdog started" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/supervisor.log | tail -2
echo "=== is each bridge's parent the supervisor watchdog, or an ad-hoc launcher? ==="
for b in $(pgrep -f _dashboard_bridge.py); do
  ppid=$(ps -o ppid= -p "$b" | tr -d ' ')
  pcmd=$(ps -o cmd= -p "$ppid" 2>/dev/null | head -c 80)
  echo "bridge $b  <- parent $ppid : $pcmd"
done
