#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$PWD/packages/hexgnn/python:$PWD/packages/hexo_engine/python:$PWD/packages/hexo_utils/python:$PWD/packages/hexo_runner/python:$PWD/packages/hexo_train/python:$PWD/packages/hexo_models/python"
python _i32_h2d_bench.py
