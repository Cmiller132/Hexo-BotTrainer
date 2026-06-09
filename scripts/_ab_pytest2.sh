#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export CUDA_VISIBLE_DEVICES=""   # force CPU so a CUDA-init test can't segfault the isolation run
cd "$ROOT"
echo "############ pure-CPU TSS logic tests (fastpath ON) ############"
python -m pytest tests/test_hexgt_threats.py tests/test_hexgt_tss_injection.py tests/test_hexgt_tss_override.py tests/test_hexgt_tss_move_selection.py tests/test_hexgt_two_stone_defense.py tests/test_hexgt_injection_additive.py -q -p no:cacheprovider 2>&1 | tail -20
echo "=== rc=${PIPESTATUS[0]} ==="
echo "############ same TSS tests, short-circuit FORCED OFF ############"
HEXGT_TSS_NOSCAN_FASTPATH=0 python -m pytest tests/test_hexgt_threats.py tests/test_hexgt_tss_injection.py tests/test_hexgt_tss_override.py tests/test_hexgt_tss_move_selection.py tests/test_hexgt_two_stone_defense.py tests/test_hexgt_injection_additive.py -q -p no:cacheprovider 2>&1 | tail -12
echo "=== rc=${PIPESTATUS[0]} ==="
