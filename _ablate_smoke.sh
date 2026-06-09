#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
cd "$ROOT" || exit 11
source /root/.venvs/hexgt-build/bin/activate
rm -rf "$ROOT/runs/hexgt_ablation_smoke"
python -u scripts/_rl_ablate.py --name SMOKE --device cpu --no-compile \
  --epochs 1 --games-per-epoch 3 --active 3 --vbatch 2 --visits 6 --max-actions 40 \
  --train-steps-per-epoch 4 --batch 8 --warmup 2 --group-size 3 \
  --l1-games 4 --l1-visits 6 --holdout-shards 4 \
  --out-dir "$ROOT/runs/hexgt_ablation_smoke" 2>&1 \
  | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
echo "=== json ==="
cat "$ROOT/runs/hexgt_ablation_smoke/ablate_SMOKE.json" 2>/dev/null | head -60
