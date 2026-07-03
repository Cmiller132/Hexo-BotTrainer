#!/usr/bin/env bash
# Round 3: live-like regime — virtual_batch_size=16, 192 games, deep ply cap
# (matches the live supervisor's active_limit=192 / vbs=16 / mean flush ~1k).
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_PERF_TRACE=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt
# args: ckpt visits games ply_cap flush_target arl vbs
echo "=== live-like: OFF ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 16
echo "=== live-like: triton + flex_pair ==="
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 16
echo "=== live-like: triton + flex_pair + half + rust_pack ==="
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 16
