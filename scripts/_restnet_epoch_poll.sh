#!/usr/bin/env bash
# Poll until the first epoch checkpoint (epoch_000001.pt) is written (= self-play +
# shuffle + TRAINING phase all survived, incl. training VRAM) or an error appears.
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
LOG=$(ls -1t "$RUNDIR"/train.*.out.log 2>/dev/null | head -1)
for i in $(seq 1 9); do
  ck=$(ls -1 "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | tail -1)
  if [[ -n "$ck" ]]; then
    echo "=== EPOCH CHECKPOINT WRITTEN: $(basename "$ck") ==="
    ls -la "$ck"
    grep -iE 'train|loss|epoch 1|sealbot|eval' "$LOG" 2>/dev/null | tail -6
    exit 0
  fi
  # Real-crash signal only: the driver process dies, or the supervisor logs a
  # relaunch/halt. (Calibration "out of memory" probes are caught by the OOM guard
  # and are NOT failures.)
  if ! ps -eo cmd | grep -q '[c]li.train_model'; then
    echo "=== DRIVER GONE ==="; grep -iE 'EXIT|RELAUNCH|HALT' "$RUNDIR/supervisor.log" | tail -3; tail -20 "$LOG"; exit 2
  fi
  if grep -qiE 'RELAUNCH|HALT:' "$RUNDIR/supervisor.log" 2>/dev/null; then
    echo "=== SUPERVISOR RELAUNCH/HALT ==="; tail -4 "$RUNDIR/supervisor.log"; exit 2
  fi
  # phase signal: selfplay live status + a peek at the latest sub-phase
  sp=$(python3 - "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(f"selfplay status={d.get('status')} done={d.get('completed_games')}/{d.get('games_started')} pos/s={d.get('search_positions_per_second')}")
except Exception: print("selfplay: (n/a)")
PY
)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  echo "[poll $i] $sp | GPU=$gpu"
  sleep 55
done
echo "=== STILL IN EPOCH 1 (re-poll) ==="
tail -3 "$LOG" 2>/dev/null | cut -c1-140
