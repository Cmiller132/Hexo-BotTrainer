#!/usr/bin/env bash
# strace -c the serve bench main thread for 20s: which syscalls dominate.
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_SERVE_FLEX=1 HEXFIELD_ASYNC_EVAL=1 HEXFIELD_DEFER_DECODE=1
export HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_RUST_PACK=1
export MALLOC_TRIM_THRESHOLD_=536870912 MALLOC_MMAP_THRESHOLD_=536870912 MALLOC_TOP_PAD_=134217728
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
PY=/root/.venvs/hexgt-build/bin/python
CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000072.pt

command -v strace >/dev/null || apt-get install -y strace >/dev/null 2>&1

"$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 192 60 1024 192 48 > /tmp/strace_bench.log 2>&1 &
BPID=$!
sleep 50   # past warmup
timeout 20 strace -c -f -p "$BPID" 2>/tmp/strace_out.txt || true
kill "$BPID" 2>/dev/null || true
wait "$BPID" 2>/dev/null || true
head -30 /tmp/strace_out.txt
