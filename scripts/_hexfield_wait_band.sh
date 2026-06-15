#!/bin/bash
RD=/mnt/e/Hexo-BotTrainer/runs/hexfield_soak
for i in $(seq 1 16); do
  if [ -f "$RD/diagnostics/hexfield.selfplay.epoch_000001.json" ]; then
    echo "epoch-1 selfplay diagnostic ready"
    break
  fi
  sleep 15
done
cd /mnt/e/Hexo-BotTrainer-hexgt
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_band_check.py
