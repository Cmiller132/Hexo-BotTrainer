#!/usr/bin/env bash
# Wait for the CURRENT trainer to reach epoch-3 self-play and adopt TRT (fail-loud
# means a bad adopt crashes instead). Exit 0 with the adopt line, 1 on crash/fail.
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
DI="$RD/diagnostics"
for i in $(seq 1 60); do          # up to 15 min
  out=$(ls -t "$DI"/trainer.*.out.log 2>/dev/null | head -1)
  err=$(ls -t "$DI"/trainer.*.err.log 2>/dev/null | head -1)
  a=$(grep -h 'adopted TRT FP16' "$out" 2>/dev/null | tail -1)
  f=$(grep -hE 'NOT ADOPTED|TRTAdoptError|FATAL|Traceback \(most recent' "$out" "$err" 2>/dev/null | tail -1)
  if [ -n "$f" ]; then echo "TRT_FAIL_OR_CRASH (t+$((i*15))s): $f"; exit 1; fi
  if [ -n "$a" ]; then
    n=$(ls "$RD"/selfplay/epoch_000003_game_*.npz 2>/dev/null | wc -l)
    L=$(python3 -c "import json;d=json.load(open('$DI/dense_cnn.selfplay.live.json'));print('epoch=%s pos/s=%.1f games=%s/%s active=%s'%(d.get('epoch'),d.get('search_positions_per_second',0),d.get('games_finished'),d.get('requested_games'),d.get('active_games')))" 2>/dev/null)
    echo "TRT_ADOPTED (t+$((i*15))s, epoch3_shards=$n): $a"
    echo "LIVE: $L"
    exit 0
  fi
  sleep 15
done
echo "TIMEOUT: no TRT adopt within 15 min (still calibrating? check supervisor log)"; exit 2
