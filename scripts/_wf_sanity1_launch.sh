#!/usr/bin/env bash
# Launch the dense_cnn_restnet_sanity_1 supervisor (detached). FIRST LAUNCH:
# fresh run dir, from-scratch init (no initialize_from in the config).
# Pre-conditions: main_4 halted (flag present, GPU free), rebuild done
# (tss/fpu-capable .so installed, gate tests green 2026-06-12).
set -uo pipefail
export ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export CONFIG=$ROOT/configs/dense_cnn_restnet_sanity_1.toml
export RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_sanity_1

if [[ ! -f "$CONFIG" ]]; then echo "ABORT: config missing: $CONFIG" >&2; exit 2; fi
if ps -eo cmd | grep train_model | grep -v grep >/dev/null 2>&1; then
  echo "ABORT: a train_model process is still running" >&2; exit 2
fi
if [[ -f "$RUNDIR/supervisor.lock" ]] && kill -0 "$(cat "$RUNDIR/supervisor.lock" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: sanity_1 supervisor already running" >&2; exit 2
fi
if ls "$RUNDIR/checkpoints"/epoch_*.pt >/dev/null 2>&1; then
  echo "NOTE: existing checkpoints found — supervisor will RESUME, not fresh-start"
fi

mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_dc_restnet_supervise_main1.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched sanity_1 supervisor (detached) pid=$!"
sleep 6
echo "--- supervisor.log ---"
tail -5 "$RUNDIR/supervisor.log" 2>/dev/null
echo "--- driver.pid ---"
cat "$RUNDIR/driver.pid" 2>/dev/null
