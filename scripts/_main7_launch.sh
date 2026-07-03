#!/usr/bin/env bash
# Launch hexfield_main_7: swap the enabled supervisor from main_6 to main_7
# (single GPU — they must not run together), start, and report first status.
# main_6 keeps all state at epoch 75 and can be re-enabled at any time.
set -eu
systemctl disable hexfield-supervisor-6 2>/dev/null || true
systemctl stop hexfield-supervisor-6 2>/dev/null || true
systemctl enable hexfield-supervisor-7
systemctl start hexfield-supervisor-7
sleep 15
systemctl is-active hexfield-supervisor-7
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7
sleep 15
LOG=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1 || true)
echo "supervisor log:"
tail -3 "$RUN/supervisor.log" 2>/dev/null || true
echo "train log: ${LOG:-none yet}"
[ -n "$LOG" ] && tail -5 "$LOG"
PID=$(pgrep -f cli.train_model | head -1 || true)
echo "trainer pid: ${PID:-NONE}"
if [ -n "${PID:-}" ]; then
  tr '\0' '\n' < "/proc/$PID/environ" \
    | grep -E 'HEXFIELD_(CHANNELS|ATTENTION_HEADS|TRUNK|TRITON_ATTN|TRITON_CONV_LN|TRITON_CONV|SERVE_HALF|CONV_FP8|COPY_STREAM|TRAIN_COMPILE)' \
    | sort
fi
echo LAUNCHED
