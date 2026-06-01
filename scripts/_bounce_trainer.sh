#!/usr/bin/env bash
# One-time trainer bounce at the epoch-1 boundary: epoch_000001.pt exists, so the
# supervisor will resume epoch 2 from it (with the new live-pos/s selfplay code).
# Kill only the trainer python (from trainer_wsl.pid); the supervisor stays up and
# relaunches. Uptime >> 180s so this is NOT a fast-crash vs the breaker.
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
PIDF="$D/diagnostics/trainer_wsl.pid"
pid="$(cat "$PIDF" 2>/dev/null | tr -d '[:space:]')"
echo "trainer_wsl.pid=$pid"
if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
  echo "no live trainer pid found via pidfile; trying pgrep"
  pid="$(pgrep -f 'train_model .*dense_cnn_model1_target_96x6' | head -1)"
  echo "pgrep pid=$pid"
fi
if [ -z "$pid" ]; then echo "ABORT: could not find trainer pid"; exit 1; fi
echo "checkpoint before bounce:"; ls -l "$D/checkpoints/epoch_000001.pt" 2>/dev/null
kill -TERM "$pid" 2>/dev/null; echo "sent SIGTERM to $pid"
for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || { echo "trainer $pid exited after ${i}s"; break; }; sleep 1; done
if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; echo "SIGKILL $pid"; sleep 2; fi
echo "post-bounce procs:"; ps -eo pid,etime,cmd 2>/dev/null | grep -E 'train_model|supervise' | grep -v grep
echo "DONE: supervisor should now relaunch (resume_from epoch_000001.pt -> epoch 2 with live-posps code)"
