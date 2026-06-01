#!/usr/bin/env bash
# Sample trainer RSS + MemAvailable + self-play progress every 30s for ~16 min.
# Verifies the post-compaction run PLATEAUS in RAM instead of climbing to the
# 28 GB ceiling like the pre-fix run (which hit 27 GB by ~66 games).
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_64x4
DI="$RD/diagnostics"
INTERVAL="${INTERVAL:-30}"
COUNT="${COUNT:-32}"
for i in $(seq 1 "$COUNT"); do
  tp=$(pgrep -f 'train_model .*64x4' | head -1)
  if [ -z "$tp" ]; then echo "$(date -u +%T) TRAINER GONE (check for OOM/crash)"; break; fi
  rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$tp/status 2>/dev/null)
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  stage=$(tail -1 "$DI/events.jsonl" 2>/dev/null | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print(d.get('event'),d.get('payload',{}).get('stage'))" 2>/dev/null)
  live=$(python3 -c "import json;d=json.load(open('$DI/dense_cnn.selfplay.live.json'));print('fin=%s/%s act=%s pos/s=%.1f t=%.0fs'%(d.get('games_finished'),d.get('requested_games'),d.get('active_games'),d.get('search_positions_per_second',0),d.get('elapsed_seconds',0)))" 2>/dev/null)
  ckpt=$(ls "$RD"/checkpoints/epoch_*.pt 2>/dev/null | wc -l)
  echo "$(date -u +%T)  trainerRSS=${rss}MB  MemAvail=${avail}MB  ckpts=${ckpt}  stage=[$stage]  $live"
  sleep "$INTERVAL"
done
echo "=== monitor done ==="