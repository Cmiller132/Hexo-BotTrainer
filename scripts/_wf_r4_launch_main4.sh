#!/usr/bin/env bash
# Launch the dense_cnn_restnet_main_4 supervisor (detached, survives wsl.exe exit).
# Pre-conditions (verified before running this): C3/C2/C1 landed + tests green;
# seed_main3/ populated (1920 npz + 1920 json, mtimes preserved); GPU free;
# configs/dense_cnn_restnet_main_4.toml final.
set -uo pipefail
export ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export CONFIG=$ROOT/configs/dense_cnn_restnet_main_4.toml
export RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4

if [[ ! -f "$CONFIG" ]]; then echo "ABORT: config missing: $CONFIG" >&2; exit 2; fi
if ! ls "$RUNDIR/selfplay/seed_main3/"*.npz >/dev/null 2>&1; then
  echo "ABORT: seed shards missing in $RUNDIR/selfplay/seed_main3" >&2; exit 2
fi
if [[ -f "$RUNDIR/selfplay/length_ema.json" ]]; then
  echo "ABORT: length_ema.json present (would defeat temperature_length_prior=115)" >&2; exit 2
fi
if [[ -f "$RUNDIR/supervisor.lock" ]] && kill -0 "$(cat "$RUNDIR/supervisor.lock" 2>/dev/null)" 2>/dev/null; then
  echo "ABORT: supervisor already running" >&2; exit 2
fi
# GPU must be free (no train_model processes)
if ps -eo cmd | grep train_model | grep -v grep >/dev/null 2>&1; then
  echo "ABORT: a train_model process is still running" >&2; exit 2
fi

mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_dc_restnet_supervise_main1.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched main_4 supervisor (detached) pid=$!"
sleep 3
echo "--- supervisor.log ---"
tail -5 "$RUNDIR/supervisor.log" 2>/dev/null
echo "--- boot log ---"
tail -5 "$RUNDIR/supervise.boot.log" 2>/dev/null
