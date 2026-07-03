#!/usr/bin/env bash
# Post-deploy health: flags in the trainer env, log errors, live pace.
set -u
PID=$(pgrep -f cli.train_model | head -1)
echo "trainer pid: ${PID:-NONE}"
if [ -n "${PID:-}" ]; then
  tr '\0' '\n' < "/proc/$PID/environ" | grep -E 'HEXFIELD_(TRITON_CONV|FLEX_PAIR|SERVE_HALF|RUST_PACK|SERVE_FLEX|TRAIN_FLEX)' | sort
fi
LOG=$(ls -t /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/train.*.out.log | head -1)
echo "--- log: $LOG ---"
grep -cE 'CantSplit|InductorError|Traceback' "$LOG" || true
tail -5 "$LOG"
echo "--- live.json ---"
cat /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/diagnostics/hexfield.selfplay.live.json 2>/dev/null | head -c 600
