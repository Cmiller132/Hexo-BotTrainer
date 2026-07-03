#!/usr/bin/env bash
# Recover from the trainer-crash halt: clear breaker flag, wipe partial epoch,
# restart, report status.
set -eu
rm -f /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/supervisor_halted.flag
bash /mnt/e/Hexo-BotTrainer-hexgt/scripts/_wipe_partial_epoch.sh
systemctl restart hexfield-supervisor-6
sleep 10
systemctl is-active hexfield-supervisor-6
LOG=$(ls -t /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/train.*.out.log | head -1)
echo "LOG: $LOG"
sleep 20
tail -3 "$LOG"
PID=$(pgrep -f cli.train_model | head -1 || true)
echo "trainer pid: ${PID:-NONE}"
if [ -n "${PID:-}" ]; then
  tr '\0' '\n' < "/proc/$PID/environ" | grep -E 'COPY_STREAM|TRAIN_COMPILE|MALLOC_TRIM' | sort
fi
