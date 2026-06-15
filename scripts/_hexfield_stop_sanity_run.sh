#!/bin/bash
# Stop the dense_cnn_restnet_sanity_1 run (owner-authorized twice:
# 2026-06-12 22:47 "there is a run currently going, you may stop it when
# desired so that you can test on gpu", and 23:54 "continue autonomously"
# issued directly after the stop plan was laid out). The halt flag is already
# in place; this TERMs exactly three verified pids — the supervisor shell
# pair and the current trainer — nothing else (dashboard 844 and
# debug_worker 1326 untouched). No kill -9.
set -u
for PID in 12667 12670; do
  if [ -d "/proc/$PID" ]; then kill "$PID" 2>/dev/null && echo "TERM supervisor $PID"; fi
done
sleep 1
if [ -d /proc/285672 ]; then kill 285672 2>/dev/null && echo "TERM trainer 285672"; fi
sleep 8
echo "--- gpu after stop:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ps -o pid,cmd -p 12667,12670,285672 2>/dev/null | tail -3
