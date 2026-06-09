#!/usr/bin/env bash
# Run the 3-config exploration ablation set (C1 derived-baseline, C2
# higher-exploration, C3 lower-exploration) sequentially. Each is a short
# BC-seeded RL run; a crash in one does not stop the others. Run via the Bash
# tool's run_in_background so the harness keeps the WSL VM alive.
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
cd "$ROOT" || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
# Reduce fragmentation; the chunked forward already caps peak VRAM ~3-4GB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="$ROOT/runs/hexgt_ablation"
mkdir -p "$OUT"
MASTER="$OUT/ablate_master.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$MASTER"; }

# Concurrency-refill: games_per_epoch (96) > active (64) so finished games are
# REPLACED and the GPU batch stays ~64-full through the epoch instead of draining
# 48->1 at the tail. L1 trimmed to 40 games. Keeps the GPU saturated -> ~13+ pos/s.
COMMON="--epochs 5 --games-per-epoch 96 --active 64 --vbatch 64 --visits 96 --max-actions 240 --train-steps-per-epoch 200 --batch 128 --lr 2e-4 --warmup 200 --l1-games 40 --l1-visits 96 --n 3 --out-dir $OUT"

run_cfg(){
  local name="$1"; shift
  log "START $name : $*"
  local t0=$(date +%s)
  python -u scripts/_rl_ablate.py --name "$name" $COMMON "$@" >"$OUT/run_$name.out" 2>"$OUT/run_$name.err" \
    && log "DONE $name ($(( ($(date +%s)-t0)/60 )) min)" \
    || log "FAILED $name (exit $?) — see run_$name.err"
}

log "=== ABLATION SET START (C1/C2/C3) ==="
# C1 derived-baseline
run_cfg C1 --total-alpha 6.6 --eps 0.25 --root-policy-temperature 1.0 --c-puct 1.5 \
            --temp-initial 1.0 --temp-final 0.2 --temp-decay 30 --temp-floor 0.1
# C2 higher-exploration
run_cfg C2 --total-alpha 9.0 --eps 0.35 --root-policy-temperature 1.0 --c-puct 2.0 \
            --temp-initial 1.2 --temp-final 0.3 --temp-decay 45 --temp-floor 0.15
# C3 lower-exploration
run_cfg C3 --total-alpha 4.5 --eps 0.15 --root-policy-temperature 1.0 --c-puct 1.0 \
            --temp-initial 1.0 --temp-final 0.1 --temp-decay 20 --temp-floor 0.05
log "=== ABLATION SET COMPLETE ==="
