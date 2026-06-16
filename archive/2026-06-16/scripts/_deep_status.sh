#!/usr/bin/env bash
set -uo pipefail
cd /mnt/e/Hexo-BotTrainer
echo "###### PROCS ######"
pgrep -af "supervise_target|train_model|bootstrap_dense_cnn|_run_bootstrap_96x8|_bootstrap_96x8|python" | grep -v "pgrep" || echo "(none)"
echo
echo "###### GPU ######"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
echo
echo "###### 96x8 run dir tree ######"
find runs/dense_cnn_model1_target_96x8 -maxdepth 3 -type f 2>/dev/null | head -40 || echo "(no 96x8 run dir)"
echo
echo "###### 96x8 bootstrap.log tail ######"
tail -n 30 runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.log 2>/dev/null || echo "(no bootstrap.log)"
echo "###### 96x8 bootstrap.rc ######"
cat runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.rc 2>/dev/null || echo "(no rc -> still running or never wrote)"
echo
echo "###### 64x4 supervisor halt/done flags + latest trainer err ######"
ls -la runs/dense_cnn_model1_target_64x4/diagnostics/supervisor_halted.flag runs/dense_cnn_model1_target_64x4/diagnostics/supervisor_completed.flag 2>/dev/null || echo "(no 64x4 flags)"
ls -t runs/dense_cnn_model1_target_64x4/diagnostics/watch_wsl.jsonl 2>/dev/null | head -1
tail -n 2 runs/dense_cnn_model1_target_64x4/diagnostics/watch_wsl.jsonl 2>/dev/null || true
