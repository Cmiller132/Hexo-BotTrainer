#!/usr/bin/env bash
# Wrapper: run an analysis python script under the training venv with the
# dense_cnn_restnet package on the path. CPU-only, niced, read-only.
set -uo pipefail
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/dense_cnn_restnet/python
export CUDA_VISIBLE_DEVICES=""
PY=/root/.venvs/hexgt-build/bin/python
exec nice -n 19 "$PY" "$@"
