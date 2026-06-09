#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
cd "$ROOT"

TSS="tests/test_hexgt_threats.py tests/test_hexgt_tss_injection.py tests/test_hexgt_tss_override.py tests/test_hexgt_tss_move_selection.py tests/test_hexgt_two_stone_defense.py tests/test_hexgt_injection_additive.py tests/test_hexgt_d6.py tests/test_hexgt_value_readout.py tests/test_hexgt_vcf.py tests/test_hexgt_new_features.py"

echo "############ (1) FULL hexgt suite — fastpath ON (default) ############"
python -m pytest tests/ -q -k "hexgt" -p no:cacheprovider 2>&1 | tail -25
echo "=== full rc=${PIPESTATUS[0]} ==="

echo "############ (2) TSS subset — short-circuit FORCED OFF (legacy full scan) ############"
HEXGT_TSS_NOSCAN_FASTPATH=0 python -m pytest $TSS -q -p no:cacheprovider 2>&1 | tail -15
echo "=== noscan-off rc=${PIPESTATUS[0]} ==="
