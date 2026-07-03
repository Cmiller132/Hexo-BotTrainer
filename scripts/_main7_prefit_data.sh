#!/usr/bin/env bash
# Build the main_7 BC-prefit dataset: symlink shards over recent main_6 sample
# epochs (same pointer-shard pattern as data/hexfield_main5_prefit, but real
# symlinks on the WSL ext4 fs). Newest epochs preferred; ~5% held out as val.
#
#   bash _main7_prefit_data.sh [N_EPOCHS (default 15)]
set -eu
SRC=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/samples
OUT="$HOME/hexfield_main7_prefit"
N_EPOCHS=${1:-15}

# Skip the newest epoch dir — the live run may still be writing it.
mapfile -t EPOCHS < <(ls -d "$SRC"/epoch_* | sort | head -n -1 | tail -n "$N_EPOCHS")
echo "using ${#EPOCHS[@]} epochs: $(basename "${EPOCHS[0]}") .. $(basename "${EPOCHS[-1]}")"

rm -rf "$OUT"
mkdir -p "$OUT/train" "$OUT/val"
i=0
for ep in "${EPOCHS[@]}"; do
  for f in "$ep"/game_*.npz; do
    i=$((i + 1))
    if [ $((i % 20)) -eq 0 ]; then d=val; else d=train; fi
    ln -s "$f" "$OUT/$d/shard_$(printf '%06d' $i).npz"
  done
done
echo "train: $(ls "$OUT/train" | wc -l) shards, val: $(ls "$OUT/val" | wc -l) shards -> $OUT"
