#!/usr/bin/env bash
# Lean status — NO in-WSL nvidia-smi / find (those hung earlier). pgrep + file checks only.
cd /mnt/e/Hexo-BotTrainer
echo "MARK_START"
echo "## 64x4 procs ##"
pgrep -af 'supervise_target_64x4_wsl.sh|train_model .*dense_cnn_model1_target_64x4' || echo "NONE_64x4"
echo "## bootstrap procs ##"
pgrep -af 'bootstrap_dense_cnn_sealbot|_bootstrap_96x8|_run_bootstrap_96x8' || echo "NONE_bootstrap"
echo "## any 96x8 supervisor/trainer ##"
pgrep -af 'supervise_target_96x8_wsl.sh|train_model .*dense_cnn_model1_target_96x8' || echo "NONE_96x8"
echo "## 64x4 watchdog last line + mtime ##"
stat -c '%y' runs/dense_cnn_model1_target_64x4/diagnostics/watch_wsl.jsonl 2>/dev/null || echo "(no watch file)"
tail -n 1 runs/dense_cnn_model1_target_64x4/diagnostics/watch_wsl.jsonl 2>/dev/null || echo "(no watch tail)"
echo "## 96x8 prefit exists? ##"
ls -la runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt 2>/dev/null || echo "NO_PREFIT"
echo "## 96x8 bootstrap rc ##"
cat runs/dense_cnn_model1_target_96x8/diagnostics/bootstrap.rc 2>/dev/null || echo "NO_RC"
echo "MARK_END"
