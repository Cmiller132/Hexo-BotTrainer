#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer
export VIRTUAL_ENV=/root/.venvs/hexo-bottrainer-wsl
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export HEXO_TRT=1
export HEXO_BUCKET_PAD_MULTIPLE=16
"$VIRTUAL_ENV/bin/python" "$ROOT/scripts/_sample_size_probe.py" 2>&1 | grep -vE 'TracerWarning|UserWarning|if x.shape|frombuffer|default stream'
