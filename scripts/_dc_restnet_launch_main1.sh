#!/usr/bin/env bash
# Launch dense_cnn_restnet_main1 — the faithful ResTNet run (96ch / 8 blocks,
# blocks_type R_R_R_T_R_R_T_R, SDPA full attention) warm-started from the HF
# human-corpus prefit, at 256 sims (owner directive 2026-06-09). Uses the dense_cnn
# config-driven CLI via a lean setsid'd supervisor (_dc_restnet_supervise_main1.sh).
# Run dir under /mnt/e/Hexo-BotTrainer/runs so the :8080 dashboard renders it
# directly (no bridge). NO watchers.
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export CONFIG="$ROOT/configs/dense_cnn_restnet_main1.toml"
export RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
export SEALBOT_PATH=/mnt/e/SealBot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUNDIR"
if [[ -f "$RUNDIR/supervisor_halted.flag" ]]; then
  echo "halt flag present at $RUNDIR/supervisor_halted.flag — clear it to launch."; exit 1
fi

setsid bash -c "tr -d '\r' < $ROOT/scripts/_dc_restnet_supervise_main1.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched dense_cnn_restnet_main1 supervisor pid=$! (detached, 96x8 ResTNet @ 256; init from HF prefit). RUNDIR=$RUNDIR"
sleep 2
echo "--- boot log ---"; tail -6 "$RUNDIR/supervise.boot.log" 2>/dev/null
