#!/usr/bin/env bash
# Launch hexgnn_rl_main1 — the FROM-SCRATCH run of the optimized transformer-free
# lineage (td96/gnn2 + D6 steerable direction edges, ~200k params; sparse graph =
# CONTEXT+window nodes removed; pipeline-overlap + Stage-1/2 perf). Owner launch
# config (2026-06-05): from scratch (NO pretrain), active=256, games/epoch=512,
# visits=1024, NO PCR. Detached via setsid so it survives the wsl.exe session.
# ADDITIVE: isolated RUNDIR (runs/hexgnn_rl_main1); no other run is touched.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgnn_rl_main1"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=60
export GAMES_PER_EPOCH=512
export EVAL_EVERY=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FROM SCRATCH (random init, no --bc-seed): the driver builds the model from the
# arch args and stamps it into every checkpoint. active=256 / games=512 / visits=1024
# / NO PCR per the owner. Everything else as proven: n=4 candidates, steerable=4 D6
# direction edges, lambda=0 hard-z, policy-surprise ON (driver default), opp-mask
# carried, KataGo move-temperature (halflife=33 floor=0.3), Dirichlet eps=0.30, and
# a fresh-warmup LR with a from-scratch-anchored exponential decay (flat 2e-4 for the
# first ~10 epochs while it learns, then halve every 20 epochs to a 2.5e-5 floor).
export EXTRA_ARGS="--from-scratch --token-dim 96 --gnn-layers 2 --attention-heads 4 \
--value-pma-seeds 2 --steerable-channels 4 --n 4 \
--active 256 --vbatch 16 --visits 1024 --max-actions 512 \
--pcr --pcr-full-proportion 0.5 --pcr-fast-visits 256 \
--train-steps-per-epoch 512 --batch 128 --lr 2e-4 --warmup 200 \
--lr-decay --lr-decay-start-step 5120 --lr-decay-halflife-steps 10240 --lr-min 2.5e-5 \
--replay-window-epochs 8 --replay-pool-cap 500000 --replay-recency-decay 0.9 \
--eval-games 40 --eval-visits 512 --eval-max-actions 1024 --eval-opening-moves 10 --eval-opening-temperature 0.6 \
--total-alpha 6.6 --eps 0.25 --root-policy-temperature 1.0 --c-puct 1.5 \
--temperature 1.0 --temperature-floor 0.3 --temperature-halflife 22 --forced-playout-k 2.0 \
--widening-max-children 96 --soft-z-lambda 0"

mkdir -p "$RUNDIR"

# Read-only dashboard bridge for the new run (mirror -> :8080 dashboard at
# /mnt/e/Hexo-BotTrainer/runs/hexgnn_rl_main1). Single-instance: only start if not
# already running. NO watchers/routines — this is the only helper we start.
BRIDGE="$ROOT/_dashboard_bridge_hexgnn.py"
if [[ -f "$BRIDGE" ]] && ! pgrep -f "_dashboard_bridge_hexgnn.py" >/dev/null 2>&1; then
  PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_runner/python" \
    setsid "$VENV/bin/python" -u "$BRIDGE" >"$RUNDIR/bridge.log" 2>&1 &
  echo "started dashboard bridge -> $RUNDIR/bridge.log"
fi
# Debug-tab note: start/point the :8080 dashboard with
#   HEXO_DEBUG_RUN_ROOT=/mnt/e/Hexo-BotTrainer-hexgt
# so the Debug tab resolves runs/hexgnn_rl_main1/checkpoints (the bridge mirror has
# the diagnostics but not the .pt checkpoints).

setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise_hexgnn.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched hexgnn_rl_main1 supervisor pid=$! (detached, FROM SCRATCH). RUNDIR=$RUNDIR"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -8
