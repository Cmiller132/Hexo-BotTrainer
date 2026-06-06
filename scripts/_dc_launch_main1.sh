#!/usr/bin/env bash
# Launch dense_cnn_rl_main1 — the FROM-SCRATCH Model 1 (96ch x 8block) baseline at
# 512 sims, for the head-to-head vs hexgnn (owner directive 2026-06-06). Uses the
# dense_cnn config-driven CLI (hexo_train.cli.train_model) via a lean setsid'd
# supervisor (_dc_supervise_main1.sh). Run dir under /mnt/e/Hexo-BotTrainer/runs so
# the :8080 dashboard renders it directly (no bridge). NO watchers.
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export CONFIG="$ROOT/configs/dense_cnn_rl_main1.toml"
export RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
export SEALBOT_PATH=/mnt/e/SealBot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUNDIR"
if [[ -f "$RUNDIR/supervisor_halted.flag" ]]; then
  echo "halt flag present at $RUNDIR/supervisor_halted.flag — clear it to launch."; exit 1
fi

setsid bash -c "tr -d '\r' < $ROOT/scripts/_dc_supervise_main1.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched dense_cnn_rl_main1 supervisor pid=$! (detached, FROM SCRATCH 96x8 @ 512). RUNDIR=$RUNDIR"
sleep 2
echo "--- boot log ---"; tail -6 "$RUNDIR/supervise.boot.log" 2>/dev/null
