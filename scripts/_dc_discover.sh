#!/usr/bin/env bash
# READ-ONLY discovery of the live dense_cnn_rl_main1 run (no kills, no writes).
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== supervisor.lock (PGID candidate) ==="
cat "$RUNDIR/supervisor.lock" 2>/dev/null || echo "(no lock file)"
echo
echo "=== driver.pid ==="
cat "$RUNDIR/driver.pid" 2>/dev/null || echo "(no driver.pid)"
echo
echo "=== halt flag present? ==="
if [[ -f "$RUNDIR/supervisor_halted.flag" ]]; then
  echo "HALT FLAG EXISTS:"; cat "$RUNDIR/supervisor_halted.flag"
else
  echo "NO halt flag (run is live)"
fi
echo "=== ps: supervisor + driver + any train_model ==="
ps -eo pid,ppid,pgid,etime,cmd | grep -E '_dc_supervise_main1|cli\.train_model' | grep -v grep || echo "(none found)"
echo "=== nvidia compute-apps ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
echo "=== nvidia mem summary ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
echo "=== latest 3 epoch checkpoints ==="
ls -1t "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | head -3 || echo "(no checkpoints)"
echo "=== supervisor.log tail ==="
tail -6 "$RUNDIR/supervisor.log" 2>/dev/null
echo "=== latest train log tail ==="
ls -1t "$RUNDIR"/train.*.out.log 2>/dev/null | head -1 | xargs -r tail -8
