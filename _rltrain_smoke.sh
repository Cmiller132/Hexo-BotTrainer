#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
cd "$ROOT" || exit 11
source /root/.venvs/hexgt-build/bin/activate
rm -rf "$ROOT/runs/hexgt_rl_smoke2"
python -u scripts/_rl_train.py --out-dir "$ROOT/runs/hexgt_rl_smoke2" \
  --epochs 1 --games-per-epoch 3 --active 3 --vbatch 4 --visits 6 --max-actions 40 \
  --train-steps-per-epoch 3 --batch 8 --warmup 1 --replay-window-epochs 2 \
  --total-alpha 6.6 --eps 0.25 --root-policy-temperature 1.0 --c-puct 1.5 \
  --temperature 1.0 --final-temperature 0.2 --temperature-decay-moves 10 \
  --eval-every 99 --device cpu --no-compile 2>&1 \
  | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
echo "=== log ==="; cat "$ROOT/runs/hexgt_rl_smoke2/rl_train.log" 2>/dev/null
echo "=== ckpt ==="; ls "$ROOT/runs/hexgt_rl_smoke2/checkpoints" 2>/dev/null
