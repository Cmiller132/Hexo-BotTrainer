#!/usr/bin/env bash
# REVERT the ep8-10 quarantine: ckpt12 amplitude probe showed the value head
# recovering under C1 alone (std 0.24-0.28 vs trough 0.14-0.21), so the
# pre-committed rule says restore the shards and stay on the patient rail.
set -uo pipefail
R=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
Q=$R/quarantine_flood_ep8_10
n=0
for f in "$Q"/epoch_*.npz "$Q"/epoch_*.json; do
  [ -e "$f" ] || continue
  mv "$f" "$R/selfplay/"
  n=$((n+1))
done
echo "restored $n files"
for ep in 08 09 10; do
  c=$(find "$R/selfplay" -maxdepth 1 -name "epoch_0000${ep}_game_*.npz" 2>/dev/null | wc -l)
  echo "ep$ep npz in selfplay: $c"
done
echo "left in quarantine: $(find "$Q" -name '*.npz' 2>/dev/null | wc -l) npz"
mv "$Q/README.txt" "$Q/README_reverted.txt" 2>/dev/null || true
echo "reverted 2026-06-12: ckpt12 amplitude probe showed recovery under C1 alone (std 0.24-0.28 vs ckpt10 trough 0.14-0.21 on identical games); quarantine unnecessary per pre-committed decision rule" >> "$Q/README_reverted.txt"
