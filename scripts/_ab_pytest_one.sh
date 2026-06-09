#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
cd "$ROOT"
echo "### threats (pure CPU), 120s timeout ###"
timeout 120 python -m pytest tests/test_hexgt_threats.py -v -p no:cacheprovider 2>&1 | tail -25
echo "=== rc=${PIPESTATUS[0]} (124=timeout) ==="
