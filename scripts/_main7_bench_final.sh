#!/usr/bin/env bash
# Final launch-stack forward numbers at c=192 (attn + tuned conv_ln, fp16).
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python
export HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3 HEXFIELD_TRUNK=CCACCACCACCACCA
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_TRITON_ATTN=1 HEXFIELD_TRITON_CONV_LN=1
"$PY" scripts/_hexfield_main6_profile.py half 2>/dev/null | grep -E "^N=|torch |ms/fwd"
