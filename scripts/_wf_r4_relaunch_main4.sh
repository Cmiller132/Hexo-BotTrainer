#!/usr/bin/env bash
# RESUME-relaunch the dense_cnn_restnet_main_4 supervisor after the 2026-06-12
# WSL VM termination (driver+supervisor died ~18:26Z with ep15 selfplay done,
# train pass not started). Unlike _wf_r4_launch_main4.sh this is a RESUME path:
# no seed-shard / length_ema preconditions (both legitimately exist mid-run) —
# the supervisor itself finds epoch_000014.pt and injects resume_from.
set -uo pipefail
export ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export CONFIG=$ROOT/configs/dense_cnn_restnet_main_4.toml
export RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4

if [[ ! -f "$CONFIG" ]]; then echo "ABORT: config missing: $CONFIG" >&2; exit 2; fi
if [[ -f "$RUNDIR/supervisor_halted.flag" ]]; then
  echo "ABORT: halt flag present — clear deliberately before relaunching" >&2; exit 2
fi
if [[ -f "$RUNDIR/supervisor.lock" ]] && kill -0 "$(cat "$RUNDIR/supervisor.lock" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: supervisor already running" >&2; exit 2
fi
if ps -eo cmd | grep train_model | grep -v grep >/dev/null 2>&1; then
  echo "ABORT: a train_model process is still running" >&2; exit 2
fi
latest=$(ls -1 "$RUNDIR/checkpoints"/epoch_*.pt 2>/dev/null | sort -V | tail -1)
echo "will resume from: ${latest:-NONE (would be a fresh start — abort if unexpected)}"
if [[ -z "$latest" ]]; then echo "ABORT: no checkpoint found" >&2; exit 2; fi

setsid bash -c "tr -d '\r' < $ROOT/scripts/_dc_restnet_supervise_main1.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "relaunched main_4 supervisor (detached) pid=$!"
sleep 5
echo "--- supervisor.log tail ---"
tail -4 "$RUNDIR/supervisor.log" 2>/dev/null
echo "--- driver.pid ---"
cat "$RUNDIR/driver.pid" 2>/dev/null
