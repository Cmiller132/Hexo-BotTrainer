#!/usr/bin/env bash
set -euo pipefail
SRC=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/samples
DST=/mnt/e/hexgt-katago/tests/katago_buffer/_scratch/p5/samples
rm -rf "$DST"
total=0
for ep in epoch_000001 epoch_000002 epoch_000003; do
  mkdir -p "$DST/$ep"
  n=0
  for f in "$SRC/$ep"/game_*.npz; do
    [ -e "$f" ] || continue
    side="${f%.npz}.json"
    [ -e "$side" ] || continue
    cp -p "$f" "$DST/$ep/"
    cp -p "$side" "$DST/$ep/"
    n=$((n+1))
    total=$((total+1))
    if [ "$n" -ge 30 ]; then break; fi
  done
  echo "$ep copied $n"
done
echo "TOTAL_NPZ $(find "$DST" -name 'game_*.npz' | wc -l)"
