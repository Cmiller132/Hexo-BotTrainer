#!/bin/bash
# Kill any prior prefit (ours), then run the step micro-bench on the GPU.
OLD=$(ps aux | grep '[_]hexfield_prefit.py' | grep -v grep | awk '{print $2}')
if [ -n "$OLD" ]; then
  echo "killing prefit: $OLD"
  kill $OLD 2>/dev/null
  sleep 5
fi
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/.venvs/hexgt-build/bin/python scripts/_hexfield_bench_step.py
