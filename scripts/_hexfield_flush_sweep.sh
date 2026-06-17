#!/usr/bin/env bash
# Throughput sweep validating the flush_target/concurrency root-cause fix.
# Tests (games, flush_target, active_root_limit) combos. flush_target ~60% of
# in-flight (games*vb4); arl >= games (hard reject guard). PLY_CAP short so each
# run is bounded; `timeout` catches hangs. Reports dec/s, evals/s, mean_flush,
# peak VRAM. expandable_segments matches production memory behavior.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1/checkpoint_epoch2.pt
PY=/root/.venvs/hexgt-build/bin/python
PLYCAP=20
# games flush_target arl
COMBOS=(
  "32 256 64"
  "32 128 64"
  "64 192 64"
  "96 256 96"
  "128 384 128"
)
for combo in "${COMBOS[@]}"; do
  set -- $combo; G=$1; FT=$2; ARL=$3
  echo "############### games=$G flush_target=$FT arl=$ARL (in-flight=$((G*4))) ###############"
  timeout 260 env PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 "$G" "$PLYCAP" "$FT" "$ARL" 2>&1 \
    | grep -E 'DECISIONS/S|EVALS/S|mean_flush|peak VRAM|Error|Traceback' \
    || echo "  >>> TIMEOUT/HANG or no output (>260s)"
  echo
done
echo "SWEEP DONE"
