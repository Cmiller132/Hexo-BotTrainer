#!/usr/bin/env bash
# Launch the hexgnn RL run (transformer-free hexgt lineage) from a pretrained seed,
# detached via setsid so it survives the transient wsl.exe session. Mirrors
# scripts/_rl_launch_main3.sh but targets the hexgnn driver/supervisor + run dir.
#
# ADDITIVE: isolated RUNDIR (runs/hexgnn_rl); the live hexgt_rl_main3 run is
# untouched. Edit the knobs below to taste; the defaults mirror model-3's run-knob
# parity minus the (removed) STV-weight arg.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgnn_rl"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=60
export GAMES_PER_EPOCH=512
export EVAL_EVERY=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BC_SEED="$RUNDIR/pretrain/hexgnn_pretrain.pt"

# model-3-parity run knobs, MINUS --short-term-value-weight (no STV in hexgnn).
export EXTRA_ARGS="--bc-seed $BC_SEED \
--active 128 --vbatch 16 --visits 1024 --max-actions 512 \
--pcr --pcr-full-proportion 0.5 --pcr-fast-visits 170 \
--train-steps-per-epoch 512 --batch 128 --lr 2e-4 --warmup 200 --replay-window-epochs 8 \
--replay-pool-cap 500000 --replay-recency-decay 0.9 \
--eval-games 40 --eval-visits 512 --eval-max-actions 1024 --eval-opening-moves 10 --eval-opening-temperature 0.6 \
--n 3 --total-alpha 6.6 --eps 0.30 --root-policy-temperature 1.0 --c-puct 1.5 \
--temperature 1.0 --temperature-floor 0.3 --temperature-halflife 33 --forced-playout-k 2.0 \
--widening-max-children 96 \
--soft-z-lambda 0"

if [[ ! -f "$BC_SEED" ]]; then echo "ABORT: pretrained seed missing: $BC_SEED (run scripts/_pretrain_hexgnn.py first)" >&2; exit 2; fi
mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise_hexgnn.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched hexgnn supervisor pid=$! (detached). RUNDIR=$RUNDIR seed=$BC_SEED"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -8
