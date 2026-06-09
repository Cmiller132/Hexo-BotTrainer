#!/usr/bin/env bash
# One-shot: hard-stop the main3 supervisor + driver + monitors. Walks the driver's
# parent chain to find the supervisor loop bash, kills the whole process group.
set +e

driver=$(pgrep -f "_rl_train.py --out-dir .*hexgt_rl_main3" | head -1)
if [ -n "$driver" ]; then
  echo "driver pid=$driver"
  # walk ppid chain up to (but not including) pid 1, collecting ancestors
  cur=$driver
  chain="$driver"
  for _ in 1 2 3 4 5 6; do
    pp=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')
    [ -z "$pp" ] && break
    [ "$pp" = "1" ] && break
    chain="$chain $pp"
    cur=$pp
  done
  echo "ancestor chain: $chain"
  pgid=$(ps -o pgid= -p "$driver" 2>/dev/null | tr -d ' ')
  echo "driver pgid=$pgid"
  [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null
  for p in $chain; do kill -9 "$p" 2>/dev/null; done
fi

# belt-and-suspenders: kill any lingering monitors/driver by name
pkill -9 -f "_rl_train.py" 2>/dev/null
pkill -9 -f "_dashboard_bridge.py" 2>/dev/null
pkill -9 -f "query-gpu=timestamp,temperature" 2>/dev/null
# the supervisor loop bash reads this script path on its cmdline only for the
# OUTER bash -c; kill those too
for p in $(pgrep -f "_rl_supervise.sh"); do kill -9 "$p" 2>/dev/null; done

sleep 2
echo "=== remaining ==="
ps -eo pid,etime,cmd | grep -iE "rl_train|dashboard_bridge|rl_supervise|query-gpu=timestamp" | grep -v grep || echo NONE
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
