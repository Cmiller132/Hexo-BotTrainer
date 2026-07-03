#!/usr/bin/env bash
# One-shot main_7 health/status snapshot.
set -u
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7
echo "supervisor: $(systemctl is-active hexfield-supervisor-7)"
ls "$RUN"/supervisor_halted.flag 2>/dev/null && echo HALT-FLAG || true
LOG=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
echo "train log: ${LOG:-none}"
if [ -n "${LOG:-}" ]; then
  grep -iE "initialized_from|warm_start|source.*prefit" "$LOG" | head -3
  grep -cE "Traceback" "$LOG" || true
  tail -4 "$LOG"
fi
PID=$(pgrep -f "cli.train_model" | head -1 || true)
echo "trainer pid: ${PID:-NONE}"
if [ -n "${PID:-}" ]; then
  tr '\0' '\n' < "/proc/$PID/environ" \
    | grep -E "HEXFIELD_(FLEX_PAIR|RUST_PACK|CONV_FP8|SERVE_FLEX)=|MALLOC_TRIM" | sort
fi
ls "$RUN"/diagnostics/epoch_*.json 2>/dev/null | tail -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
