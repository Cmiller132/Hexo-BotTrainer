#!/bin/bash
# Run a hexfield smoke config through hexo_train, full output to stdout + log.
set -uo pipefail
CONFIG="${1:-configs/hexfield_smoke_tiny.toml}"
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="packages/hexfield/python:packages/dense_cnn_restnet/python"
LOG="runs/_smoke_run.log"
/root/.venvs/hexgt-build/bin/python -u -m hexo_train.cli.train_model "$CONFIG" 2>&1 | tee "$LOG"
echo "PYTHON_EXIT=${PIPESTATUS[0]}"
