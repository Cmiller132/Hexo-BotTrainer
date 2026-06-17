#!/usr/bin/env bash
# Halt-flag-guarded bounce of dense_cnn_restnet_main1 to pick up the lowered
# shuffle_min_rows. Kills the supervisor group, verifies clean, then relaunches
# (re-reads config, resumes from latest checkpoint).
set -uo pipefail
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
HALT="$RUNDIR/supervisor_halted.flag"

echo "bounce 2026-06-09: lower shuffle_min_rows 100000->8000" > "$HALT"
echo "wrote halt flag"

PGID=$(ps -eo pid,pgid,cmd | grep '_dc_restnet_supervise' | grep -v grep | awk '{print $2}' | head -1)
echo "supervisor PGID=${PGID:-<none>}"
[[ -n "${PGID:-}" ]] && { kill -KILL -"$PGID" 2>/dev/null && echo "killed group -$PGID"; }
sleep 3

echo "=== survivors (expect none) ==="
ps -eo pid,cmd | grep -E '_dc_restnet_supervise|cli.train_model' | grep -v grep || echo "ZERO survivors"
echo "=== GPU (expect freed) ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | head -1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "=== latest checkpoint (resume target) ==="
ls -1t "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | head -1

rm -f "$HALT" "$RUNDIR/supervisor.lock"
echo "cleared halt flag + stale lock"

echo "=== relaunch ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_restnet_launch_main1.sh | bash
