#!/bin/bash
for i in $(seq 1 120); do
  if ! pgrep -f _head_to_head.py >/dev/null 2>&1; then
    echo "=== EVAL PROCESS EXITED ==="
    echo "vs_dense games: $(ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_h2h/vs_dense/*.hxr 2>/dev/null | wc -l)"
    echo "vs_sealbot games: $(ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_h2h/vs_sealbot/*.hxr 2>/dev/null | wc -l)"
    exit 0
  fi
  sleep 10
done
echo "STILL RUNNING after wait"
