#!/usr/bin/env bash
# Poll until training ENGAGES (shuffle fires + train pass + a real loss) for the
# post-bounce epoch, confirming the lowered shuffle_min_rows worked. ~8 min window.
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
for i in $(seq 1 9); do
  # shuffle fired this epoch?
  shuf=$(ls -1dt "$RUNDIR"/shuffleddata/*epoch_0000* 2>/dev/null | head -1)
  # a train diagnostic with a loss?
  trainjson=$(ls -1t "$RUNDIR"/diagnostics/*train*.json 2>/dev/null | head -1)
  loss=""
  if [[ -n "$trainjson" ]]; then
    loss=$(grep -oE '"loss":[ ]*[0-9.]+' "$trainjson" 2>/dev/null | head -1)
  fi
  # train event in events.jsonl
  ev=$(grep -oE '"phase":[ ]*"train"[^}]*"loss":[ ]*[0-9.]+|"train_loss":[ ]*[0-9.]+|"status":[ ]*"trained"' "$RUNDIR/events.jsonl" 2>/dev/null | tail -1)
  ck2=$(ls -1 "$RUNDIR/checkpoints"/epoch_000002.pt 2>/dev/null)
  if [[ -n "$loss" || -n "$ev" ]]; then
    echo "=== TRAINING ENGAGED ==="
    echo "train json: $trainjson"
    [[ -n "$trainjson" ]] && tr -d '\n' < "$trainjson" | head -c 500; echo
    echo "event: $ev"
    exit 0
  fi
  sp=$(python3 - "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(f"selfplay ep={d.get('epoch')} done={d.get('completed_games')}/{d.get('games_started')} pos/s={round(d.get('search_positions_per_second',0),1)}")
except Exception: print("selfplay n/a")
PY
)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  echo "[poll $i] $sp | shuffle=$([[ -n "$shuf" ]] && echo yes || echo no) ck2=$([[ -n "$ck2" ]] && echo yes || echo no) | GPU=$gpu"
  sleep 55
done
echo "=== still in epoch self-play / pre-train (re-poll) ==="
