#!/usr/bin/env bash
# Thread-pool probe: OMP off / passive, rayon sizing — on top of malloc tunables.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export MALLOC_TRIM_THRESHOLD_=536870912 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TOP_PAD_=134217728
export HEXFIELD_PERF_TRACE=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt

echo "=== OMP_NUM_THREADS=1 ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48
echo "=== OMP passive wait ==="
OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48
echo "=== OMP=1 + RAYON_NUM_THREADS=8 ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=8 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48
