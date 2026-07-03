#!/usr/bin/env bash
# Copy-stream pinned staging: parity + throughput.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export MALLOC_TRIM_THRESHOLD_=536870912 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TOP_PAD_=134217728
export HEXFIELD_PERF_TRACE=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt
export HEXFIELD_REF_RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000071.pt

echo "--- save fresh full-stack baseline (copy_stream OFF) ---"
"$PY" scripts/_hexfield_serve_ref.py save /tmp/main6_cs_ref.npz
echo "--- parity: + copy_stream (must be ~bit-identical, tol 1e-6) ---"
HEXFIELD_COPY_STREAM=1 "$PY" scripts/_hexfield_serve_ref.py check /tmp/main6_cs_ref.npz 1e-6

echo "=== throughput: vbs=48 + copy_stream ==="
HEXFIELD_COPY_STREAM=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48
