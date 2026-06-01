#!/usr/bin/env bash
# Full status: all relevant procs + GPU + bootstrap log/rc.
set -uo pipefail
cd /mnt/e/Hexo-BotTrainer
echo "=== ALL relevant procs (supervisor/trainer/bootstrap) ==="
pgrep -af "supervise_target|train_model|bootstrap_dense_cnn|_run_bootstrap_96x8|_bootstrap_96x8" || echo "(none)"
echo
echo "=== nvidia-smi (WSL view) ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1 || echo "nvidia-smi failed"
echo
echo "=== 96x8 bootstrap log tail ==="
tail -n 20 runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.log 2>/dev/null || echo "(no bootstrap log)"
echo "=== 96x8 bootstrap rc ==="
cat runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.rc 2>/dev/null || echo "(no rc / still running)"
echo
echo "=== 64x4 trainer stderr tail (is it alive/recent?) ==="
ls -t runs/dense_cnn_model1_target_64x4/diagnostics/trainer.*.err.log 2>/dev/null | head -1 | xargs -r tail -n 5 2>/dev/null || echo "(none)"
