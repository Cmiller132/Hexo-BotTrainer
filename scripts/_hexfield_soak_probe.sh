#!/bin/bash
RD=/mnt/e/Hexo-BotTrainer/runs/hexfield_soak
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== shards written so far (epoch 1 self-play in progress) ==="
find "$RD/samples" -name '*.npz' 2>/dev/null | wc -l
echo "=== main-thread stack (2 samples) ==="
MAIN=$(ps aux | grep 'hexfield_soak.toml' | grep -v grep | grep -v 'bash -c' | awk '{print $2}' | head -1)
echo "pid: ${MAIN:-NONE}"
for i in 1 2; do
  /root/.venvs/hexfield-dev/bin/py-spy dump --pid "$MAIN" 2>/dev/null | grep -A8 MainThread | head -10
  echo "---"
  sleep 3
done
