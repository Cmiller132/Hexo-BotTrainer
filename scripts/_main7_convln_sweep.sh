#!/usr/bin/env bash
# Sweep conv+LN kernel tiles at c=192 (quiet GPU).
set -eu
PY=/root/.venvs/hexgt-build/bin/python
S=/mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_convln_sweep.py
for bm in 32 64; do
  for w in 4 8; do
    for st in 2 3; do
      HEXFIELD_CONVLN_BM=$bm HEXFIELD_CONVLN_WARPS=$w HEXFIELD_CONVLN_STAGES=$st "$PY" "$S"
    done
  done
done
echo SWEEP DONE
