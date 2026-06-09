#!/usr/bin/env bash
# Launch the unattended BC-seeded RL run via the supervisor, DETACHED with setsid
# so it survives the transient wsl.exe session (per the build-isolation notes).
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgt_rl"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=20
export GAMES_PER_EPOCH=96
export EVAL_EVERY=2
# Real-run knobs: active=64 keeps the leaf batches full (the 1.5->13 pos/s lever),
# games/epoch=96 refills so concurrency stays ~64, eval at matched visits=200.
export EXTRA_ARGS="--active 64 --vbatch 64 --visits 128 --max-actions 240 --train-steps-per-epoch 300 --batch 128 --lr 2e-4 --warmup 200 --replay-window-epochs 8 --eval-games 24 --eval-visits 200 --n 3"
mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched supervisor pid=$! (detached). RUNDIR=$RUNDIR"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -5
