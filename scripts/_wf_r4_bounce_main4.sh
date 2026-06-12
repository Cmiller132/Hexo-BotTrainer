#!/usr/bin/env bash
# Bounce the main_4 DRIVER only (supervisor stays up and relaunches with
# _resume_config.toml rebuilt from the just-edited CONFIG -> C1 knee=110 active,
# resume_from latest checkpoint). Driver uptime >> 300s so this is not a
# fast-crash for the breaker. No pkill; exact pid only.
set -u
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
DRV=$(cat "$D/driver.pid" 2>/dev/null | tr -d '[:space:]')
SUP=$(cat "$D/supervisor.lock" 2>/dev/null | tr -d '[:space:]')
echo "driver=$DRV supervisor=$SUP"
if [ -z "$DRV" ] || ! kill -0 "$DRV" 2>/dev/null; then echo "ABORT: no live driver"; exit 1; fi
if [ -z "$SUP" ] || ! kill -0 "$SUP" 2>/dev/null; then echo "ABORT: supervisor not alive (would not relaunch)"; exit 1; fi
kill -TERM "$DRV" && echo "TERM sent to driver $DRV"
for i in $(seq 1 30); do
  if ! kill -0 "$DRV" 2>/dev/null; then echo "driver exited after ${i}s"; break; fi
  sleep 1
done
if kill -0 "$DRV" 2>/dev/null; then kill -KILL "$DRV"; echo "SIGKILL driver"; fi
sleep 8
echo "--- supervisor.log tail ---"
tail -5 "$D/supervisor.log"
echo "--- resume config knee check ---"
grep -E "length_decay_moves_left_knee|resume_from" "$D/_resume_config.toml" 2>/dev/null
echo "--- new driver ---"
cat "$D/driver.pid" 2>/dev/null
ps -eo pid,etime,cmd | grep train_model | grep -v grep
