#!/usr/bin/env bash
# Clean STOP of the live dense_cnn_rl_main1 run (owner directive 2026-06-09):
# set halt flag (resumable marker), then SIGKILL the whole setsid process group so
# the supervisor cannot relaunch the driver. Does NOT delete checkpoints/run dir.
set -uo pipefail
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
HALT="$RUNDIR/supervisor_halted.flag"

# Re-derive the supervisor's process-group id (PGID) from the live process.
PGID=$(ps -eo pid,pgid,cmd | grep '_dc_supervise_main1' | grep -v grep | awk '{print $2}' | head -1)
echo "supervisor PGID = ${PGID:-<none found>}"

# 1) Halt flag FIRST so any in-flight relaunch is blocked.
echo "owner stop 2026-06-09: switch GPU to dense_cnn_restnet run. Completed through epoch 40 (epoch_000040.pt). Resumable: clear this flag + bash scripts/_dc_launch_main1.sh." > "$HALT"
echo "wrote halt flag: $HALT"

# 2) Kill the whole process group (supervisor + inner bash + driver python + children).
if [[ -n "${PGID:-}" ]]; then
  kill -KILL -"$PGID" 2>/dev/null && echo "sent SIGKILL to group -$PGID" || echo "group kill returned nonzero (may already be gone)"
else
  echo "no PGID found; nothing to group-kill"
fi
sleep 3

# 3) Verify zero survivors.
echo "=== survivors (expect none) ==="
ps -eo pid,pgid,etime,cmd | grep -E '_dc_supervise_main1|cli\.train_model' | grep -v grep || echo "ZERO survivors"
echo "=== any SealBot/eval stragglers from this run ==="
ps -eo pid,cmd | grep -iE 'sealbot' | grep -v grep || echo "no sealbot stragglers"

# 4) Verify GPU freed.
echo "=== nvidia compute-apps (expect none for this run) ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
echo "=== nvidia mem/util ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null

# 5) Record final state.
echo "=== final checkpoint ==="
ls -1t "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | head -1
echo "=== halt flag in place ==="
ls -la "$HALT"
