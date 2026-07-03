#!/usr/bin/env bash
# Round 4: scheduler knobs under the deployed serve stack.
# vbs sweep + depth2 + complete-overlap at live-like settings.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export HEXFIELD_PERF_TRACE=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt
# args: ckpt visits games ply_cap flush_target arl vbs

echo "=== vbs=16 (deployed baseline) ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 16
echo "=== vbs=32 ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 32
echo "=== vbs=48 ==="
"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48
echo "=== vbs=32 + depth2 ==="
HEXFIELD_PIPELINE_DEPTH2=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 32
echo "=== vbs=32 + depth2 + complete_overlap ==="
HEXFIELD_PIPELINE_DEPTH2=1 HEXFIELD_PIPELINE_COMPLETE_OVERLAP=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 32
