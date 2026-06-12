#!/usr/bin/env bash
R=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
for ep in 06 07 08 09 10 11 12 13; do
  c=$(find "$R/selfplay" -maxdepth 1 -name "epoch_0000${ep}_game_*.npz" 2>/dev/null | wc -l)
  echo "ep$ep: $c"
done
echo "seed: $(find "$R/selfplay/seed_main3" -name '*.npz' 2>/dev/null | wc -l)"
echo "quarantined npz: $(find "$R/quarantine_flood_ep8_10" -name '*.npz' 2>/dev/null | wc -l)"
echo "README: $(ls "$R/quarantine_flood_ep8_10/README.txt" 2>/dev/null || echo MISSING)"
