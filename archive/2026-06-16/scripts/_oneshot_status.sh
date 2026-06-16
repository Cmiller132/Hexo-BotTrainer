#!/usr/bin/env bash
echo "NOW=$(date -u +%FT%TZ)"
echo "=== PGREP 64x4 sup ==="; pgrep -af "supervise_target_64x4_wsl.sh" || echo NONE
echo "=== PGREP 64x4 trainer ==="; pgrep -af "train_model .*dense_cnn_model1_target_64x4" || echo NONE
echo "=== PGREP 96x8 sup ==="; pgrep -af "supervise_target_96x8_wsl.sh" || echo NONE
echo "=== PGREP 96x8 trainer ==="; pgrep -af "train_model .*dense_cnn_model1_target_96x8" || echo NONE
echo "=== PGREP bootstrap ==="; pgrep -af "bootstrap_dense_cnn_sealbot" || echo NONE
echo "=== ckpts dir ==="; ls -la /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/ 2>&1 || echo NO_DIR
echo "=== bootstrap.log tail ==="; tail -n 20 /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.log 2>&1 || echo NO_LOG
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1
echo "ALLDONE"
