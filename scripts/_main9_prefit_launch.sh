#!/usr/bin/env bash
# Clean-launch the main_9 BC prefit (c=128 / heads 2 / trunk CCACCACCA env is
# LOAD-BEARING — the prefit checkpoint must be built at the main_9 arch so it
# tensor-loads into the supervisor net). Mirrors _main7_prefit_launch.sh; only
# kills prior PREFIT processes.
# Data is the TRIMMED main_8 ep7 samples (fast rows already stripped) staged at
# $DATA/{train,val}. Prefit trains eagerly (TRAIN_FLEX stays off — the prefit
# perf note from the main_4 playbook still applies) and doesn't touch the serve
# kernels. 4 BC epochs -> checkpoint_epoch3.pt is the warm start.
set -u
REPO=/mnt/e/Hexo-BotTrainer-hexgt
DATA=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_9_prefit
OUT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_9_prefit
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

test -d "$DATA/train" || { echo "no prefit data — stage trimmed main_8 ep7 samples into $DATA/{train,val} first"; exit 1; }
mkdir -p "$OUT"
cd $REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HEXFIELD_SUPPORT_RADIUS=4
export HEXFIELD_CHANNELS=128
export HEXFIELD_ATTENTION_HEADS=2
export HEXFIELD_TRUNK=CCACCACCA
setsid nohup /root/.venvs/hexgt-build/bin/python scripts/_hexfield_prefit.py \
  --data "$DATA" --out "$OUT" --epochs 4 \
  --workers 6 --device cuda > "$LOG" 2>&1 < /dev/null &
sleep 3
NEW=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}' | head -1)
echo "prefit pid: ${NEW:-FAILED}  log: $LOG"
