#!/usr/bin/env bash
# Focused concurrency test: flush_target=1024 (never the trigger -> batch size is
# purely select-driven), so mean_flush reveals whether bigger active-game pools
# actually produce bigger GPU batches. PLY_CAP=12 keeps decision count modest;
# 480s timeout distinguishes "slow but scaling" from "deadlocked/collapsing".
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1/checkpoint_epoch2.pt
PY=/root/.venvs/hexgt-build/bin/python
PLYCAP=12
for G in 32 64 96; do
  echo "############### games=$G ft=1024 arl=$G (in-flight=$((G*4)), max decisions=$((G*PLYCAP))) ###############"
  START=$(date +%s)
  timeout 480 env PYTHONPATH=packages/hexfield/python:packages/dense_cnn_restnet/python \
    "$PY" scripts/_hexfield_selfplay_throughput.py "$CKPT" 512 "$G" "$PLYCAP" 1024 "$G" 2>&1 \
    | grep -E 'decisions=|DECISIONS/S|EVALS/S|mean_flush|peak VRAM|Error|Traceback'
  rc=${PIPESTATUS[0]}
  echo "  elapsed=$(( $(date +%s) - START ))s rc=$rc $([ $rc -eq 124 ] && echo '(TIMEOUT)')"
  echo
done
echo "FOCUSED DONE"
