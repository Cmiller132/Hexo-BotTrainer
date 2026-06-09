#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
echo "############ BEFORE: legacy full threats() scan (NOSCAN_FASTPATH=0) ############"
HEXGT_TSS_NOSCAN_FASTPATH=0 python scripts/_perf_tss_ab2.py --games 64 --moves 10 --visits 512 --vbatch 64 --tag BEFORE 2>&1 | grep -E "BEFORE|PERF_AB2|Error|Traceback" | tail -8
echo "############ AFTER: O(1) has_threats short-circuit (NOSCAN_FASTPATH=1) ############"
HEXGT_TSS_NOSCAN_FASTPATH=1 python scripts/_perf_tss_ab2.py --games 64 --moves 10 --visits 512 --vbatch 64 --tag AFTER 2>&1 | grep -E "AFTER|PERF_AB2|Error|Traceback" | tail -8
echo "ALL_DONE"
