#!/usr/bin/env bash
# Seed main_4's replay window with main_3 ep1-5 shards (= ckpt5's exact training
# corpus, 46,576 rows). mtime-preserved (cp -p) — the shuffle window selects
# files mtime-newest-first via rglob. NO .hxr, NO ep6+ shards, NO length_ema.json.
set -euo pipefail
SRC=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_3/selfplay
DST=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4/selfplay/seed_main3
mkdir -p "$DST"
n=0
for ep in 1 2 3 4 5; do
  pre=$(printf "epoch_%06d_game_" "$ep")
  for f in "$SRC/$pre"*.npz "$SRC/$pre"*.json; do
    cp -p "$f" "$DST/"
    n=$((n+1))
  done
done
echo "copied $n files"
echo "npz: $(ls "$DST" | grep -c npz.)  json: $(ls "$DST" | grep -c json.)"
echo "newest 3 by mtime:"
ls -t "$DST" | head -3
echo "oldest 3 by mtime:"
ls -t "$DST" | tail -3
# Safety: assert nothing from ep6+ slipped in and no length_ema/hxr present
bad=$(ls "$DST" | grep -vcE "^epoch_00000[1-5]_game_.*\.(npz|json)$" || true)
echo "non-conforming files: $bad"
