#!/usr/bin/env bash
# Foreground supervisor launch for the RICHER BC-seeded RL run (config C1) — the
# intervention on the data-starved regression: 256 games/epoch + 65536 train
# samples/epoch (512 steps @ batch 128). Run via the Bash tool's run_in_background
# so the harness keeps the WSL VM alive; the supervisor loops with auto-relaunch +
# RAM watchdog + circuit breaker.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
export ROOT
export VENV=/root/.venvs/hexgt-build
export RUNDIR="$ROOT/runs/hexgt_rl_main2"
export SEALBOT_PATH=/mnt/e/SealBot
export EPOCHS=60
export GAMES_PER_EPOCH=256
export EVAL_EVERY=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Config C1 (derived baseline). active=64 < games/epoch=256 -> the loop keeps 64
# games concurrent (refilling as they finish) so the GPU batch stays full while
# in-flight RAM stays bounded to 64 games (games_per_epoch only lengthens the
# epoch, it does NOT raise peak RAM). 512 train steps = 65536 samples/epoch.
# Seed from the prior run's epoch-8 checkpoint (model + optimizer) — DELIBERATE
# user choice to watch the GNN learn from epoch-8 with the richer data, knowing it
# is weaker than the BC seed (epoch-8 = 22.5% vs dense_cnn / 0% vs SealBot, vs the
# BC seed's 45%/10%). Do NOT auto-revert to the BC seed.
# SELF-PLAY visits=512 AND EVAL visits=512 (matched 512-vs-512 head-to-head,
# dense_cnn's regime). NOTE: re-baselines eval — the new win rates at 512 visits
# are NOT directly comparable to the prior 200-visit data points (deeper search
# both sides). Self-play at 512 sims is ~4x the per-move search of 128.
export EXTRA_ARGS="--bc-seed /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/checkpoints/hexgt_rl_epoch000008.pt \
--active 64 --vbatch 64 --visits 512 --max-actions 512 \
--train-steps-per-epoch 512 --batch 128 --lr 2e-4 --warmup 200 --replay-window-epochs 8 \
--eval-games 40 --eval-visits 512 --eval-max-actions 1024 --eval-opening-moves 10 --eval-opening-temperature 0.6 \
--n 3 --total-alpha 6.6 --eps 0.25 --root-policy-temperature 1.0 --c-puct 1.5 \
--temperature 1.0 --final-temperature 0.2 --temperature-decay-moves 30 --temperature-floor 0.1 --forced-playout-k 2.0 \
--short-term-value-weight 0.10"
mkdir -p "$RUNDIR"
tr -d '\r' < "$ROOT/scripts/_rl_supervise.sh" | bash
