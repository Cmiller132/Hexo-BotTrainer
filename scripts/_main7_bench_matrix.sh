#!/usr/bin/env bash
# main_7 (c=192, heads 3x64, trunk CCAx5) forward bench matrix. Needs a QUIET
# GPU — stop hexfield-supervisor-6 first. Random weights (kernel timing only;
# honest pos/s comes from the post-prefit e2e).
#
#   1 flex-d64   arch-only baseline: stock flex @ d=64, triton conv, fp16 serve
#   2 +attn      bespoke fused attention kernel
#   3 +ln        + conv+LN epilogue fusion
#   4 +fp8       + e4m3 trunk convs (numerics pending arena eval)
#   5 heads6-d32 the "naive heads" counterfactual (6 heads, d=32, stock flex)
#   6 main6-ref  c=128 live stack reference (cross-check vs known numbers)
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
export HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1

M7="HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA"
BASE="HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1"

run() {
  local tag="$1"; shift
  echo; echo "===== $tag ====="
  env "$@" "$PY" scripts/_hexfield_main6_profile.py half
}

run flex-d64    $M7 $BASE
run +attn       $M7 $BASE HEXFIELD_TRITON_ATTN=1
run +ln         $M7 $BASE HEXFIELD_TRITON_ATTN=1 HEXFIELD_TRITON_CONV_LN=1
run +fp8        $M7 $BASE HEXFIELD_TRITON_ATTN=1 HEXFIELD_TRITON_CONV_LN=1 HEXFIELD_CONV_FP8=1
run heads6-d32  HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=6 HEXFIELD_TRUNK=CCACCACCACCACCA $BASE
run main6-ref   HEXFIELD_CHANNELS=128 $BASE HEXFIELD_TRITON_ATTN=1
echo; echo MATRIX DONE
