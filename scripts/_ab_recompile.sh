#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_LOGS=recompiles
cd "$ROOT"
# 4 moves over 64 games; count torch.compile recompiles + the reasons.
python scripts/_perf_tss_ab2.py --games 64 --moves 4 --visits 512 --vbatch 64 --tag RECOMP 2>recomp.err.log 1>recomp.out.log || true
echo "=== pos/s line ==="; grep -E "RECOMP\]|EVAL split" recomp.out.log
echo "=== recompile count ==="; grep -c "Recompiling" recomp.err.log || echo 0
echo "=== first recompile reasons (head) ==="; grep -A2 "Recompiling function" recomp.err.log | head -40
echo "ALL_DONE"
