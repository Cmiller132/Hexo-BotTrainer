#!/usr/bin/env bash
# Stop driver + supervisor + ANY running bridge (so the new supervisor's watchdog
# becomes the single bridge owner). Gentle: TERM by PID/pattern, never wsl.exe.
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py _dashboard_bridge.py; do
  for p in $(pgrep -f "$pat"); do echo "TERM $pat pid $p"; kill "$p" 2>/dev/null; done
done
sleep 4
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py _dashboard_bridge.py; do
  for p in $(pgrep -f "$pat"); do echo "KILL $pat pid $p"; kill -9 "$p" 2>/dev/null; done
done
sleep 2
echo "=== survivors ==="
pgrep -af "_rl_train.py|_rl_supervise|_rl_run_fg|_dashboard_bridge.py" | grep -v pgrep || echo "all stopped"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
