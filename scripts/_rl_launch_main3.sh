#!/usr/bin/env bash
# Launch the model-3 RL run (PMA value head + 4 GNN layers + full TSS/soft-Z
# stack) from epoch 0, seeded by the Phase-C pretrained checkpoint, at MAIN2-PARITY
# run settings (read from runs/.../_rl_run_fg.sh). Detached via setsid so it
# survives the transient wsl.exe session.
#
# RUN-KNOB PARITY WITH main2 (only the seed + the model-3 arch/stack differ):
#   256 games/epoch, visits 512 (self-play AND eval), max-actions 512,
#   512 train steps x batch 128 = 65,536 samples/pass, 500k replay pool cap,
#   recency decay 0.9, widening-max-children 96, eval-every 3, eval-games 40,
#   eval-visits 512, STV horizons [4,12,24] @ weight 0.10, total-alpha 6.6,
#   epochs 60, dead-cell candidate rule (code-level in candidates.rs), soft_z=0.5.
# ISOLATED RUNDIR; the halted lineage (hexgt_rl, hexgt_rl_main2) is untouched.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgt_rl_main3"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=60
export GAMES_PER_EPOCH=256
export EVAL_EVERY=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BC_SEED="$RUNDIR/pretrain/hexgt_model3_pretrain.pt"
# main2 EXTRA_ARGS verbatim, except --bc-seed (model-3 pretrain, not main's e8) and
# explicit --replay-pool-cap/--replay-recency-decay (main2 used the same defaults).
export EXTRA_ARGS="--bc-seed $BC_SEED \
--active 64 --vbatch 64 --visits 512 --max-actions 512 \
--train-steps-per-epoch 512 --batch 128 --lr 2e-4 --warmup 200 --replay-window-epochs 8 \
--replay-pool-cap 500000 --replay-recency-decay 0.9 \
--eval-games 40 --eval-visits 512 --eval-max-actions 1024 --eval-opening-moves 10 --eval-opening-temperature 0.6 \
--n 3 --total-alpha 6.6 --eps 0.25 --root-policy-temperature 1.0 --c-puct 1.5 \
--temperature 1.0 --final-temperature 0.2 --temperature-decay-moves 30 --temperature-floor 0.1 --forced-playout-k 2.0 \
--widening-max-children 96 --short-term-value-weight 0.10"

if [[ ! -f "$BC_SEED" ]]; then echo "ABORT: pretrained seed missing: $BC_SEED" >&2; exit 2; fi
mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched model-3 supervisor pid=$! (detached). RUNDIR=$RUNDIR seed=$BC_SEED"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -8
