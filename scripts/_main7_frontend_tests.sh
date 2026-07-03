#!/usr/bin/env bash
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=packages/hexo_frontend/python:packages/hexo_engine/python:packages/hexo_utils/python:packages/hexo_runner/python:packages/hexo_train/python:packages/hexo_models/python:packages/dense_cnn_restnet/python
/root/.venvs/hexgt-build/bin/python -m pytest tests/test_hexo_runner_match_mode.py tests/test_sealbot_adapter.py -x -q 2>&1 | tail -15
