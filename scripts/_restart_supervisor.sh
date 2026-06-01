#!/usr/bin/env bash
# Stop the running (buggy-in-memory) supervisor + its epoch-1-redo trainer so a
# fresh supervisor (with the fixed set_resume) can relaunch and resume epoch 2
# from epoch_000001.pt. Kill the supervisor FIRST so it can't relaunch the trainer.
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
echo "BEFORE:"; ps -eo pid,cmd | grep -E 'supervise_target_96x6_wsl|train_model .*target_96x6' | grep -v grep
pkill -f 'supervise_target_96x6_wsl.sh' && echo "SIGTERM supervisor" || echo "no supervisor matched"
sleep 3
pkill -f 'train_model .*dense_cnn_model1_target_96x6' && echo "SIGTERM trainer" || echo "no trainer matched"
sleep 3
# hard-kill any survivors
pkill -9 -f 'supervise_target_96x6_wsl.sh' 2>/dev/null
pkill -9 -f 'train_model .*dense_cnn_model1_target_96x6' 2>/dev/null
sleep 1
rm -f "$D/diagnostics/supervisor_wsl.lock"; echo "lock removed"
echo "AFTER:"; ps -eo pid,cmd | grep -E 'supervise_target_96x6_wsl|train_model .*target_96x6' | grep -v grep || echo "ALL STOPPED"
echo "epoch_000001.pt (resume target):"; ls -l "$D/checkpoints/epoch_000001.pt"
