#!/usr/bin/env bash
# Wait for the k=2 trainer (32750, epoch 2) to COMPLETE enough fresh games so the
# realized opening-move diversity (.hxr) is measurable. The diversity signal lives
# in played moves, not the pruned npz target. Reference = a marker at "now"; count
# epoch_000002 game npz (re)written after it. Also abort on crash/HALT.
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
REF=/tmp/k2_ref_marker
touch "$REF"
ERR=$(ls -t "$D"/diagnostics/trainer.20260530_025514.err.log 2>/dev/null)
for i in $(seq 1 60); do          # ~60*30s = 30 min
  halt=$(ls "$D"/diagnostics/supervisor_halted.flag 2>/dev/null)
  if [ -n "$halt" ]; then echo "HALT FLAG"; cat "$halt"; exit 1; fi
  tb=$(grep -hiE 'Traceback|panicked|CUDA error|TRTAdoptError' "$ERR" 2>/dev/null | tail -2)
  if [ -n "$tb" ]; then echo "TRAINER ERROR:"; echo "$tb"; exit 1; fi
  fresh=$(find "$D"/selfplay -name 'epoch_000002_game_*.npz' -newer "$REF" 2>/dev/null | wc -l)
  if [ "$fresh" -ge 40 ]; then
    echo "K2_GAMES_READY (t+$((i*30))s): fresh epoch-2 games since marker = $fresh"
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT: only $(find "$D"/selfplay -name 'epoch_000002_game_*.npz' -newer "$REF" 2>/dev/null | wc -l) fresh games"; exit 1
