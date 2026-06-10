#!/usr/bin/env bash
# Confirm the resume config knobs, then poll until epoch 3 finishes and report
# whether the TRAIN phase engaged (status + loss + steps) vs skipped.
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== resume config knobs ==="
grep -E 'train_samples_per_epoch' "$RUNDIR/_resume_config.toml"
grep -E 'shuffle_min_rows' "$RUNDIR/_resume_config.toml"
echo "=== polling for epoch 3 train result ==="
for i in $(seq 1 9); do
  if [[ -f "$RUNDIR/diagnostics/epoch_000003.json" ]]; then
    echo "=== epoch_000003.json train result ==="
    /root/.venvs/hexgt-build/bin/python - "$RUNDIR/diagnostics/epoch_000003.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
hits=[]
def walk(o,path=""):
    if isinstance(o,dict):
        keys=set(o.keys())
        if ("status" in keys and ("loss" in keys or "steps" in keys or "train_bucket_level" in keys or "reason" in keys)):
            hits.append((path,{k:o[k] for k in ("status","reason","steps","loss","losses","available_rows","requested","train_bucket_level","train_steps_since_last_reload") if k in o}))
        for k,v in o.items(): walk(v,path+"/"+k)
    elif isinstance(o,list):
        for j,v in enumerate(o): walk(v,f"{path}[{j}]")
walk(d)
for p,h in hits: print(p,"->",h)
PY
    exit 0
  fi
  sp=$(/root/.venvs/hexgt-build/bin/python - "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1])); print(f"ep={d.get('epoch')} done={d.get('completed_games')}/{d.get('games_started')} pos/s={round(d.get('search_positions_per_second',0),1)}")
except Exception: print("n/a")
PY
)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
  echo "[poll $i] selfplay $sp | GPU=$gpu"
  sleep 55
done
echo "=== epoch 3 not done yet (re-poll) ==="
