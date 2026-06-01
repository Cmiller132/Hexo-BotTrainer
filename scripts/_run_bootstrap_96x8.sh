#!/usr/bin/env bash
# Wrapper: run the 96x8 bootstrap, tee output to a log, and record rc to a sentinel.
set -uo pipefail

cd /mnt/e/Hexo-BotTrainer
LOG=runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.log
mkdir -p "$(dirname "$LOG")"

echo "=== bootstrap start $(date -Is) ===" | tee "$LOG"
set +e
bash scripts/_bootstrap_96x8.sh >>"$LOG" 2>&1
rc=$?
set -e
echo "=== bootstrap end $(date -Is) rc=$rc ===" | tee -a "$LOG"
echo "$rc" > runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.rc
