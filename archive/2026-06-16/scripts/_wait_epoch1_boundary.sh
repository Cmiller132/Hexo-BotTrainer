#!/usr/bin/env bash
# Wait for the epoch-1 checkpoint to appear and stabilize (so a resume has a
# clean point), then report the authoritative epoch-1 self-play pos/s. Exits 0
# when ready so the harness notifies the operator to bounce the trainer into
# epoch 2 (which then runs the new live-pos/s self-play code).
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
CK="$D/checkpoints/epoch_000001.pt"
SP="$D/diagnostics/dense_cnn.selfplay.epoch_000001.json"
for i in $(seq 1 2400); do          # ~10h safety bound at 15s/iter
  if [ -f "$CK" ]; then
    s1=$(stat -c %s "$CK" 2>/dev/null); sleep 4; s2=$(stat -c %s "$CK" 2>/dev/null)
    if [ -n "$s1" ] && [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]; then
      echo "EPOCH1_CHECKPOINT_READY $(date -u +%FT%TZ)  size=$s2"
      ls -l "$CK"
      python3 - "$SP" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print("EPOCH1 search_pos_s=%.1f  full_pos_s=%.1f  searched=%d  games_finished=%d/%d  selfplay_elapsed=%.0fs"
          %(d.get("search_positions_per_second",0),d.get("positions_per_second",0),
            d.get("searched_positions",0),d.get("games_finished",0),d.get("requested_games",0),
            d.get("elapsed_seconds",0)))
except Exception as e:
    print("epoch1 selfplay diag not readable yet:",e)
PY
      cat "$D/diagnostics/trainer_wsl.pid" 2>/dev/null | sed 's/^/TRAINER_PID /'
      exit 0
    fi
  fi
  sleep 15
done
echo "TIMEOUT: epoch1 checkpoint did not appear"
exit 1
