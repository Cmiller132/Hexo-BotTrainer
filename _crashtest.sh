#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
cd "$ROOT"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
echo "=== crash-recovery tests ==="
python -m pytest tests/test_hexgt_crash_recovery.py -q 2>&1 | tail -20
echo "=== full suite ==="
python -m pytest tests -q 2>&1 | tail -6
