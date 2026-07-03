#!/usr/bin/env bash
# Confirm the main_7 trainer warm-started from the BC prefit checkpoint.
set -u
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7
LOG=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
echo "--- checkpoint-related lines in train log:"
grep -inE "checkpoint|prefit|initialize|warm" "$LOG" | grep -v UserWarning | head -10
echo "--- run dir:"
ls "$RUN" | head -20
echo "--- any stage diagnostics:"
ls -t "$RUN"/diagnostics/ 2>/dev/null | head -5
