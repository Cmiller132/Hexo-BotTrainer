#!/usr/bin/env bash
# Launch the NEW model-3 RL run (PMA value head + 4 GNN layers + full TSS/soft-Z
# stack) from epoch 0, seeded by the Phase-C pretrained checkpoint. Detached via
# setsid so it survives the transient wsl.exe session (build-isolation notes).
#
# ISOLATED: its own RUNDIR (runs/hexgt_rl_main3); does NOT touch the halted
# lineage (runs/hexgt_rl, hexgt_rl_main2 keep their checkpoints/buffers + halt
# flags). The supervisor's halt flag is PER-RUNDIR, so main3 has none -> it runs.
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgt_rl_main3"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=40
export GAMES_PER_EPOCH=96
export EVAL_EVERY=2
BC_SEED="$RUNDIR/pretrain/hexgt_model3_pretrain.pt"
# Throughput knobs (memory: active=64 is the 1.5->13 pos/s lever; refill
# games/epoch=96 keeps ~64 concurrent). soft_z_lambda defaults to 0.5 (config).
# TSS injection/forced-visit/override/move-guard + count-4 features are always-on
# in the native code; the new arch (4 GNN + PMA k=2) comes from the seed's arch.
export EXTRA_ARGS="--bc-seed $BC_SEED --active 64 --vbatch 64 --visits 128 --max-actions 240 --train-steps-per-epoch 300 --batch 128 --lr 2e-4 --warmup 200 --replay-window-epochs 8 --eval-games 24 --eval-visits 200 --n 3 --forced-playout-k 2.0 --widening-max-children 96 --total-alpha 6.6 --c-puct 1.5 --temperature 1.0 --final-temperature 0.2 --temperature-decay-moves 30"

if [[ ! -f "$BC_SEED" ]]; then echo "ABORT: pretrained seed missing: $BC_SEED" >&2; exit 2; fi
mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched model-3 supervisor pid=$! (detached). RUNDIR=$RUNDIR seed=$BC_SEED"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -8
