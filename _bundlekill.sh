#!/usr/bin/env bash
# Gently stop the hexgt RL supervisor + driver by PID (NEVER force-kill wsl.exe —
# that collateral-kills unrelated work). Kill the supervisor chain FIRST so it
# cannot relaunch the driver, then the driver.
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py; do
  for p in $(pgrep -f "$pat"); do
    echo "TERM $pat pid $p"
    kill "$p" 2>/dev/null
  done
done
sleep 4
for pat in _rl_run_fg.sh _rl_supervise.sh _rl_train.py; do
  for p in $(pgrep -f "$pat"); do
    echo "KILL $pat pid $p"
    kill -9 "$p" 2>/dev/null
  done
done
sleep 2
echo "=== survivors ==="
pgrep -af "_rl_train.py|_rl_supervise.sh|_rl_run_fg.sh" | grep -v pgrep || echo "all stopped"
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
