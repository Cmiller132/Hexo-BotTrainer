#!/usr/bin/env bash
set -eu
PID=$(pgrep -f cli.train_model | head -1)
echo "pid $PID"
tr '\0' '\n' < "/proc/$PID/environ" | grep -E 'COPY_STREAM|TRAIN_COMPILE|MALLOC_|VIRTUAL|RUST_PACK' | sort || true
grep -n 'virtual_batch_size' /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/_resume_config.toml 2>/dev/null | head -3 || true
