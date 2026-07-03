#!/usr/bin/env bash
# End-to-end self-play throughput A/B via the real continuous scheduler.
# Usage: _main6_e2e_ab.sh <visits> <games> <ply_cap> <flush_target>
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_PERF_TRACE=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt
V=${1:-512}; G=${2:-96}; P=${3:-30}; F=${4:-512}

echo "=== OFF (baseline) ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" "$V" "$G" "$P" "$F"
echo "=== triton_conv + flex_pair ==="
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" "$V" "$G" "$P" "$F"
echo "=== + serve_half ==="
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" "$V" "$G" "$P" "$F"
