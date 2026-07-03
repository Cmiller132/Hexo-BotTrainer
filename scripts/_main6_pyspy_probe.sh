#!/usr/bin/env bash
# py-spy the serve loop: start the bench, attach for 40s, dump top functions.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt

MALLOC_TRIM_THRESHOLD_=536870912 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TOP_PAD_=134217728 \
  "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48 > /tmp/pyspy_bench.log 2>&1 &
BPID=$!
sleep 45   # past compile warmup
/root/.venvs/hexgt-build/bin/py-spy record --pid "$BPID" --duration 40 --rate 75 --native \
  --format raw -o /tmp/pyspy_profile.txt 2>/tmp/pyspy_err.txt || cat /tmp/pyspy_err.txt
kill "$BPID" 2>/dev/null || true
wait "$BPID" 2>/dev/null || true
echo "--- top collapsed stacks by samples ---"
sort -t' ' -k2 -rn /tmp/pyspy_profile.txt 2>/dev/null | head -25
