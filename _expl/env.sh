#!/usr/bin/env bash
# Shared env for the read-only exploration analysis (CPU only, no GPU contention).
export ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexgnn/python:$ROOT/packages/hexo_frontend/python"
export PY=/root/.venvs/hexgt-build/bin/python
