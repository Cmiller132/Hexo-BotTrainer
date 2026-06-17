#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== train_model driver process ==="
ps -eo pid,etime,cmd | grep 'cli.train_model' | grep -v grep || echo "NO driver process"
echo "=== driver.pid file ==="
dp=$(cat "$RUNDIR/driver.pid" 2>/dev/null); echo "pid=$dp"; kill -0 "$dp" 2>/dev/null && echo "ALIVE" || echo "not-found-by-kill0"
echo "=== selfplay live (key fields) ==="
python3 - "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    for k in ("status","epoch","completed_games","games_started","active_games","raw_samples","search_positions_per_second","elapsed_seconds"):
        print(f"  {k}: {d.get(k)}")
except Exception as e:
    print("  (no live json yet:", e, ")")
PY
echo "=== last 3 events ==="
tail -3 "$RUNDIR/events.jsonl" 2>/dev/null | cut -c1-140
echo "=== checkpoints written ==="
ls -1 "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | tail -3 || echo "  (none yet — still epoch 1)"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== supervisor tail ==="
tail -2 "$RUNDIR/supervisor.log"
