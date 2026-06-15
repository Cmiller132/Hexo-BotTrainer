#!/bin/bash
# Wait up to ~5 min for a checkpoint file to appear.
CKPT="${1:-/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1/checkpoint_epoch2.pt}"
for i in $(seq 1 30); do
  if [ -f "$CKPT" ]; then
    echo "READY after $((i * 10))s"
    ls -la "$CKPT"
    exit 0
  fi
  sleep 10
done
echo "TIMEOUT"
tail -c 120 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1.log
