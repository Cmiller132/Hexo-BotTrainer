#!/usr/bin/env bash
# Quarantine main_4's un-decayed flood shards (epochs 8-10) from the training
# window. The shuffle re-scans selfplay/ (rglob *.npz, mtime-newest) at each
# epoch's train stage, so moving these OUTSIDE selfplay/ removes them from the
# next window without a bounce. Fully reversible (mv back). Rationale: ep8-10
# rows were written BEFORE the C1 length-decay activated; they are the
# hedge-teaching mass (ml80 share 0.48-0.53) still dominating the window and
# blocking value-head recovery (ep12 played at 192 dec/game by ckpt11).
set -euo pipefail
R=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
Q=$R/quarantine_flood_ep8_10
mkdir -p "$Q"
n=0
for ep in 8 9 10; do
  pre=$(printf "epoch_%06d_game_" "$ep")
  for f in "$R/selfplay/$pre"*.npz "$R/selfplay/$pre"*.json; do
    [ -e "$f" ] || continue
    mv "$f" "$Q/"
    n=$((n+1))
  done
done
cat > "$Q/README.txt" <<'EOF'
Quarantined 2026-06-12: per-game npz+json shards of main_4 epochs 8-10.
WHY: written before the C1 length-decay activated (epoch 11+); marathon flood
rows (moves_left>=80 share 0.48-0.53, hedge-teaching) dominated the training
window and blocked value-head amplitude recovery. The .hxr aggregates remain
in selfplay/ (not scanned for training). RESTORE: mv * ../selfplay/
EOF
echo "moved $n files"
echo "remaining selfplay epochs (npz):"
for ep in 6 7 8 9 10 11 12 13; do
  pre=$(printf "epoch_%06d_game_" "$ep")
  c=$(ls "$R/selfplay/$pre"*.npz 2>/dev/null | wc -l)
  echo "  ep$ep: $c"
done
echo "seed shards: $(ls "$R/selfplay/seed_main3/"*.npz 2>/dev/null | wc -l)"
echo "quarantined: $(ls "$Q"/*.npz 2>/dev/null | wc -l) npz"
