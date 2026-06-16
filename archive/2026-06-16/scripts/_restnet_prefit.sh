#!/usr/bin/env bash
# Env wrapper for the dense_cnn_restnet HF prefit/converter. Mirrors the live run's
# venv + PYTHONPATH (proven-working native resolution) and forwards all args.
# CUDA_VISIBLE_DEVICES is inherited (set empty for CPU validate; 0 for GPU prefit).
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
PY=$VENV/bin/python
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/dense_cnn_restnet/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PY" -u "$ROOT/scripts/bootstrap_dense_cnn_restnet_hf.py" "$@"
