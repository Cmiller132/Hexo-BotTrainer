#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer
export VIRTUAL_ENV=/root/.venvs/hexo-bottrainer-wsl
PY="$VIRTUAL_ENV/bin/python"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
cd "$ROOT"
"$PY" -m pytest tests/ -k dense_cnn -q 2>&1 | tail -20
