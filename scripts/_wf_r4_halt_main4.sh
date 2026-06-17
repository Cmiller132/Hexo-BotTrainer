#!/usr/bin/env bash
# Clean halt of the main_4 supervisor + driver (owner directive 2026-06-12
# 18:48Z: don't wait for the ep15 selfplay re-run). Halt flag first (blocks
# any future supervisor start until deliberately cleared), supervisor before
# driver (so the EXIT is not relaunched). Targeted pids only — never pkill.
set -uo pipefail
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4

echo "halt: owner directive 2026-06-12T18:48Z (skip ep15 selfplay re-run; resumable from epoch_000014.pt by clearing this flag)" > "$RUNDIR/supervisor_halted.flag"

spid=$(cat "$RUNDIR/supervisor.lock" 2>/dev/null || true)
dpid=$(cat "$RUNDIR/driver.pid" 2>/dev/null || true)
echo "supervisor.lock pid: ${spid:-none} | driver.pid: ${dpid:-none}"

for p in $spid; do
  if [ -n "$p" ] && ps -p "$p" > /dev/null 2>&1; then
    # Kill the supervisor's whole process group? No — targeted: the lock pid
    # plus its parent wrapper if still alive.
    ppid=$(ps -o ppid= -p "$p" | tr -d ' ')
    kill "$p" 2>/dev/null && echo "killed supervisor $p"
    if [ -n "$ppid" ] && [ "$ppid" != "1" ] && ps -o cmd= -p "$ppid" 2>/dev/null | grep -q supervise; then
      kill "$ppid" 2>/dev/null && echo "killed supervisor wrapper $ppid"
    fi
  fi
done
sleep 1
if [ -n "$dpid" ] && ps -p "$dpid" > /dev/null 2>&1; then
  kill "$dpid" 2>/dev/null && echo "sent SIGTERM to driver $dpid"
fi
# Wait for the driver to shut down (it handles SIGTERM; prior 143 exits were clean).
for i in $(seq 1 30); do
  ps -p "$dpid" > /dev/null 2>&1 || break
  sleep 2
done
ps -p "$dpid" > /dev/null 2>&1 && echo "driver STILL ALIVE after 60s" || echo "driver down"
echo "--- remaining run processes ---"
ps -eo pid,cmd | grep -E "[t]rain_model|[s]upervise" || echo "(none)"
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "--- supervisor.log tail ---"
tail -3 "$RUNDIR/supervisor.log"
