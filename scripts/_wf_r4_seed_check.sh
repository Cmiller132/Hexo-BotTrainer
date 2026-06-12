#!/usr/bin/env bash
set -u
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4/selfplay/seed_main3
total=$(ls "$D" | wc -l)
npz=$(ls "$D" | grep -c '\.npz$')
json=$(ls "$D" | grep -c '\.json$')
bad=$(ls "$D" | grep -vE '^epoch_00000[1-5]_game_[0-9]+\.(npz|json)$' | head -10)
badn=$(ls "$D" | grep -cvE '^epoch_00000[1-5]_game_[0-9]+\.(npz|json)$')
echo "total=$total npz=$npz json=$json nonconforming=$badn"
echo "nonconforming sample:"
echo "$bad"
echo "mtime range:"
ls -t "$D" | head -1; ls -t "$D" | tail -1
echo "per-epoch npz counts:"
for ep in 1 2 3 4 5; do
  p=$(printf "epoch_%06d" "$ep")
  echo "  ep$ep: $(ls "$D" | grep -c "^${p}_game_.*npz")"
done
