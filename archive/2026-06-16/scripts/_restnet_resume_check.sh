#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== _resume_config.toml: threshold + resume_from ==="
grep -E 'shuffle_min_rows' "$RUNDIR/_resume_config.toml"
grep -E 'resume_from' "$RUNDIR/_resume_config.toml"
echo "=== load_checkpoint (resumed from?) ==="
tr -d '\n' < "$RUNDIR/diagnostics/load_checkpoint.json" 2>/dev/null | head -c 320; echo
echo "=== selfplay live ==="
tr -d '\n' < "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" 2>/dev/null | head -c 320; echo
echo "=== driver alive ==="
ps -eo pid,etime,cmd | grep '[c]li.train_model' | head -1
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1
