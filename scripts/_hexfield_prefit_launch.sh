#!/bin/bash
# Clean-launch the hexfield BC prefit on the GPU (M3).
# Only ever kills prior PREFIT processes (exact cmdline match on this repo's
# prefit script); everything else on the GPU is reported, never touched.
set -u
REPO=/mnt/e/Hexo-BotTrainer-hexgt
LOG=$REPO/runs/hexfield_bc_1.log

OLD=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}')
if [ -n "$OLD" ]; then
  echo "killing old prefit: $OLD"
  kill $OLD 2>/dev/null
  sleep 3
fi
echo "--- gpu compute apps before launch (informational):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader

cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup /root/.venvs/hexgt-build/bin/python scripts/_hexfield_prefit.py \
  --data data/hexfield_bootstrap --out runs/hexfield_bc_1 --epochs 4 \
  --workers 6 --device cuda > "$LOG" 2>&1 < /dev/null &
sleep 3
NEW=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}' | head -1)
echo "prefit pid: ${NEW:-FAILED}"
