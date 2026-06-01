#!/usr/bin/env bash
# Wait until the 64x4 run produces its first self-play evidence (epoch_000001 shard
# + TRT adopt + live pos/s) or crashes. Exit 0 with a summary, 1 on crash, 2 timeout.
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
DI="$RD/diagnostics"
for i in $(seq 1 80); do          # up to ~20 min
  out=$(ls -t "$DI"/trainer.*.out.log 2>/dev/null | head -1)
  err=$(ls -t "$DI"/trainer.*.err.log 2>/dev/null | head -1)
  # crash / halt detection
  if [ -f "$DI/supervisor_halted.flag" ]; then echo "HALTED: $(cat "$DI/supervisor_halted.flag")"; exit 1; fi
  f=$(grep -hE 'Traceback \(most recent|FATAL|TRTAdoptError|NOT ADOPTED|CUDA error|out of memory' "$out" "$err" 2>/dev/null | tail -1)
  if [ -n "$f" ]; then echo "CRASH_SIG (t+$((i*15))s): $f"; exit 1; fi
  shard=$(ls -t "$RD"/selfplay/epoch_000001_game_*.npz 2>/dev/null | head -1)
  if [ -n "$shard" ]; then
    n=$(ls "$RD"/selfplay/epoch_000001_game_*.npz 2>/dev/null | wc -l)
    trt=$(grep -h 'adopted TRT FP16' "$out" 2>/dev/null | tail -1)
    live=$(python3 -c "import json;d=json.load(open('$DI/dense_cnn.selfplay.live.json'));print('epoch=%s status=%s pos/s=%.1f games=%s/%s active=%s'%(d.get('epoch'),d.get('status'),d.get('search_positions_per_second',0),d.get('games_finished'),d.get('requested_games'),d.get('active_games')))" 2>/dev/null)
    echo "SELFPLAY_LIVE (t+$((i*15))s): epoch1_shards=$n"
    echo "  live: $live"
    echo "  trt:  ${trt:-<not adopted yet>}"
    exit 0
  fi
  # surface current stage while waiting
  stage=$(tail -1 "$DI/events.jsonl" 2>/dev/null | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d.get('event'),d.get('payload',{}).get('stage'))" 2>/dev/null)
  echo "t+$((i*15))s stage=$stage"
  sleep 15
done
echo "TIMEOUT: no epoch1 self-play shard within ~20 min"; exit 2
