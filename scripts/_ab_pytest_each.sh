#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
cd "$ROOT"
for t in test_hexgt_tss_override test_hexgt_tss_injection test_hexgt_tss_move_selection test_hexgt_two_stone_defense test_hexgt_injection_additive test_hexgt_d6 test_hexgt_value_readout; do
  echo "### $t (90s timeout) ###"
  timeout 90 python -m pytest "tests/$t.py" -q -p no:cacheprovider 2>&1 | tail -4
  echo "--- $t rc=${PIPESTATUS[0]} (124=timeout/hang) ---"
done
