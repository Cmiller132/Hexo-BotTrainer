#!/usr/bin/env bash
# Clean-launch the main_7 BC prefit (c=192 / heads 3 / trunk CCAx5 env is
# LOAD-BEARING — the prefit checkpoint must be built at the main_7 arch).
# Mirrors _hexfield_prefit_launch.sh; only kills prior PREFIT processes.
# Prefit trains eagerly (TRAIN_FLEX stays off — the prefit perf note from the
# main_4 playbook still applies) and doesn't touch the serve kernels.
set -u
REPO=/mnt/e/Hexo-BotTrainer-hexgt
DATA="$HOME/hexfield_main7_prefit"
OUT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7_prefit
LOG=$OUT/prefit.log

OLD=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}')
if [ -n "$OLD" ]; then
  echo "killing old prefit: $OLD"
  kill $OLD 2>/dev/null
  sleep 3
fi
echo "--- gpu compute apps before launch (informational):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader

test -d "$DATA/train" || { echo "no prefit data — run _main7_prefit_data.sh first"; exit 1; }
mkdir -p "$OUT"
cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_CHANNELS=192
export HEXFIELD_ATTENTION_HEADS=3
export HEXFIELD_TRUNK=CCACCACCACCACCA
setsid nohup /root/.venvs/hexgt-build/bin/python scripts/_hexfield_prefit.py \
  --data "$DATA" --out "$OUT" --epochs 4 \
  --workers 6 --device cuda > "$LOG" 2>&1 < /dev/null &
sleep 3
NEW=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}' | head -1)
echo "prefit pid: ${NEW:-FAILED}  log: $LOG"
