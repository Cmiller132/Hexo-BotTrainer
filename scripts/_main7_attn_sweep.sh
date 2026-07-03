#!/usr/bin/env bash
set -eu
PY=/root/.venvs/hexgt-build/bin/python
S=/mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_attn_sweep.py
for cfg in "64 64 4 3" "64 64 4 2" "128 64 4 3" "64 128 4 3" "128 64 8 3" "64 64 8 2"; do
  set -- $cfg
  HEXFIELD_ATTN_BM=$1 HEXFIELD_ATTN_BN=$2 HEXFIELD_ATTN_WARPS=$3 HEXFIELD_ATTN_STAGES=$4 "$PY" "$S" \
    || echo "BM=$1 BN=$2 warps=$3 stages=$4: FAILED (resources)"
done
echo ATTN SWEEP DONE
