#!/usr/bin/env bash
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "=== resume config knobs ==="
grep -E 'games_per_epoch' "$RUNDIR/_resume_config.toml" | head -3
grep -E 'shuffle_min_rows' "$RUNDIR/_resume_config.toml"
grep -E 'train_samples_per_epoch' "$RUNDIR/_resume_config.toml"
grep -E 'resume_from' "$RUNDIR/_resume_config.toml"
grep -E 'active_games' "$RUNDIR/_resume_config.toml" | head -1
echo "=== exactly one supervisor + one driver? ==="
echo "supervisors: $(ps -eo cmd | grep -c '[_]dc_restnet_supervise')  drivers: $(ps -eo cmd | grep -c '[c]li.train_model')"
ps -eo pid,etime,cmd | grep -E '[_]dc_restnet_supervise|[c]li.train_model'
echo "=== load_checkpoint (resumed from) ==="
tr -d '\n' < "$RUNDIR/diagnostics/load_checkpoint.json" 2>/dev/null | grep -oE '"checkpoint_ref":[ ]*"[^"]+"|"status":[ ]*"[a-z]+"' | head -2
echo "=== selfplay live ==="
/root/.venvs/hexgt-build/bin/python - "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" <<'PY' 2>/dev/null
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"  epoch={d.get('epoch')} requested_games={d.get('requested_games')} started={d.get('games_started')} active={d.get('active_games')}/{d.get('active_limit')} done={d.get('completed_games')}")
except Exception as e: print("  (live json not ready:", e, ")")
PY
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1
