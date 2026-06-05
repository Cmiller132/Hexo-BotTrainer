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
#   epochs 60, dead-cell candidate rule (code-level in candidates.rs).
# SOFT-Z DISABLED 2026-06-04 (owner decision): --soft-z-lambda 0 -> pure hard
#   outcome value targets (z), threaded via the driver cfg dict (the TOML [samples]
#   section is NOT read by the driver). Fixes the soft-Z->opening-optimism feedback
#   loop confirmed by the lambda0 A/B; buffer swapped to hard-z mirror at relaunch.
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
# vbatch=128 (owner decision): measured ~1.23x pos/s vs vbatch=64. It changes the
# virtual-loss/eval-batch interaction (4 vs 8 sequential NN-eval rounds per move at
# 512 visits -> different self-play trajectories), accepted as the throughput choice.
# TSS no-scan short-circuit + featurize-forward overlap remain bit-identical.
# BACKSTOP SALVAGE 2026-06-04 (backstop watcher): training --batch 128 -> 64.
# The run halted (breaker noProgress=4) never completing epoch 0: telemetry-confirmed
# VRAM-ceiling crash at the self-play->train transition (self-play floor ~6.2GB, then
# training crests >10.8GB and dies at the ~12.3GB card limit; manifests as either
# cudaErrorMemoryAllocation or the expandable_segments "!handles_.at(i)" allocator
# assert). The in-code gc.collect()+empty_cache() at _rl_train.py:607-609 is
# insufficient. batch=64 halves the training-phase activation/grad peak (the dynamic
# GNN batch is the dominant variable-size term). REVERSIBLE; resume-safe (batch is not
# in the checkpoint). NOTE: this drops samples/pass 65,536->32,768 vs the documented
# main2 parity; to restore parity instead, owner may prefer --train-steps-per-epoch 1024.
# MOVE TEMPERATURE 2026-06-04 (owner): KataGo-style smooth exponential halflife.
# temp(ply)=0.3+(1.0-0.3)*2^(-ply/halflife). HALFLIFE 44.26 -> 33 (2nd tweak): measured
# that opening exploration is ~free (valΔ~0) but midgame temp~0.70 is slightly high;
# shorter halflife trims midgame (t(35) 0.70->0.61) while keeping the opening high
# (t(10)~0.86). Floor kept at 0.3 (owner: don't lower noise floor). Pairs with --eps
# 0.25 -> 0.30: "more noise, less temp" -- shift exploration to root Dirichlet (shapes
# policy targets / search, no z-noise at lambda=0) from played-move temperature (writes
# +/-1 outcomes). New curve: t(0)=1.0, t(80)=0.43, t(200)=0.31. --temperature-halflife
# >0 SUPERSEDES the linear --final-temperature/--temperature-decay-moves (inert fallback).
# VBATCH 2026-06-04 (owner): --vbatch 128 -> 16. Measured (vs sequential vbatch=1 on real
# positions): vbatch=128 distorted the visit policy by KL~0.69 / root value ~0.36 and changed
# the played move ~1/3 of the time; vbatch=16 is near-sequential (KL~0.08). 512/16 = 32 NN-
# feedback rounds/move (was 4). --active 64 -> 96 -> 128 compensates throughput: at 96/vbatch16
# nothing saturated (GPU ~65% util, CPU ~3.8/32 threads, self-play VRAM ~1.7GB), so more
# concurrent games enlarge the coalesced forward batch to use the GPU headroom. 128 (not 256):
# games/epoch=256 has 5-12x length variance, so active=256 (no refill) is gated by the longest
# game with idle slots draining; 128 keeps refilling. vbatch=16 keeps self-play VRAM bounded.
# VISITS + PCR 2026-06-04 (owner): --visits 512 -> 1024 AND KataGo Playout Cap Randomization
# (Wu 2020), 50% full. Per MOVE, p=0.5 -> FULL search (1024 visits, RECORDED as a training
# row, with Dirichlet noise + forced playouts + the temperature schedule); else FAST search
# (--pcr-fast-visits 170 == KataGo's 1/6 ratio at N=1024, NOT recorded, no noise, played
# greedily). Avg cost ~0.5*1024+0.5*170 = 597 visits/move (~+17% vs 512); recorded rows/epoch
# ~HALVE but each is a 1024-visit target. Exploration machinery (noise/forced-playouts/temp)
# is full-only; tactical injection/hitting-set/move-guard apply to BOTH (safety preserved).
# eval-visits stays 512 (fixed SealBot-eval yardstick for cross-epoch trend continuity). PCR
# threaded via the cfg dict (driver does NOT read TOML [selfplay]). Spec + faithful-mapping:
# docs/analysis/HEXGT_PCR_KATAGO_MAPPING.md.
export EXTRA_ARGS="--bc-seed $BC_SEED \
--active 128 --vbatch 16 --visits 1024 --max-actions 512 \
--pcr --pcr-full-proportion 0.5 --pcr-fast-visits 170 \
--train-steps-per-epoch 512 --batch 64 --lr 2e-4 --warmup 200 --replay-window-epochs 8 \
--replay-pool-cap 500000 --replay-recency-decay 0.9 \
--eval-games 40 --eval-visits 512 --eval-max-actions 1024 --eval-opening-moves 10 --eval-opening-temperature 0.6 \
--n 3 --total-alpha 6.6 --eps 0.30 --root-policy-temperature 1.0 --c-puct 1.5 \
--temperature 1.0 --temperature-floor 0.3 --temperature-halflife 33 --forced-playout-k 2.0 \
--final-temperature 0.2 --temperature-decay-moves 30 \
--widening-max-children 96 --short-term-value-weight 0.10 \
--soft-z-lambda 0"

if [[ ! -f "$BC_SEED" ]]; then echo "ABORT: pretrained seed missing: $BC_SEED" >&2; exit 2; fi
mkdir -p "$RUNDIR"
setsid bash -c "tr -d '\r' < $ROOT/scripts/_rl_supervise.sh | bash" >"$RUNDIR/supervise.boot.log" 2>&1 &
echo "launched model-3 supervisor pid=$! (detached). RUNDIR=$RUNDIR seed=$BC_SEED"
sleep 2
echo "--- boot log ---"; cat "$RUNDIR/supervise.boot.log" 2>/dev/null | tail -8
