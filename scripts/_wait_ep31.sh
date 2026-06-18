#!/usr/bin/env bash
# Bounded waiter for epoch 31 (first per-block + flex-train pass) to land.
D=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_3
for i in $(seq 1 180); do
  if [ -f "$D/diagnostics/epoch_000031.json" ] && [ -f "$D/checkpoints/epoch_000031.pt" ]; then
    echo "EP31_LANDED iter=$i"
    break
  fi
  if [ -f "$D/supervisor_halted.flag" ]; then
    echo "HALTED iter=$i"
    break
  fi
  sleep 5
done
echo "--- latest ckpt ---"
ls -t "$D/checkpoints"/epoch_*.pt | head -1
echo "--- halt? ---"
if [ -f "$D/supervisor_halted.flag" ]; then echo HALTED; else echo none; fi
