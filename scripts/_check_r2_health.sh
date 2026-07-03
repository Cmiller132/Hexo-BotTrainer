#!/usr/bin/env bash
set -u
systemctl is-active hexfield-supervisor-6
ps -eo pid,etime,args | grep train_model | grep -v grep || echo "NO TRAINER PROCESS"
LOG=$(ls -t /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/train.*.out.log | head -1)
echo "LOG: $LOG"
echo "tracebacks: $(grep -c Traceback "$LOG" 2>/dev/null || echo 0)"
tail -6 "$LOG"
PID=$(pgrep -f cli.train_model | head -1 || true)
if [ -n "${PID:-}" ]; then
  tr '\0' '\n' < "/proc/$PID/environ" | grep -E 'COPY_STREAM|TRAIN_COMPILE|MALLOC_TRIM' | sort
fi
tail -c 400 /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/diagnostics/hexfield.selfplay.live.json
