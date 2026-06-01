#!/usr/bin/env bash
# Tail the 96x8 bootstrap log + report rc sentinel (if finished).
set -uo pipefail
cd /mnt/e/Hexo-BotTrainer
LOG=runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.log
RC=runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.rc
echo "=== tail bootstrap.log ==="
tail -n 30 "$LOG" 2>/dev/null || echo "(no log yet)"
echo "=== rc ==="
if [[ -f "$RC" ]]; then echo "rc=$(cat "$RC")"; else echo "(still running)"; fi
echo "=== prefit checkpoint ==="
ls -la runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt 2>/dev/null || echo "(not yet written)"
