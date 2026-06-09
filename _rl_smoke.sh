#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
SEALBOT=/mnt/e/SealBot
cd "$ROOT" || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python:$SEALBOT:$SEALBOT/best"
export SEALBOT_PATH="$SEALBOT"
rm -rf "$ROOT/runs/hexgt_rl_smoke"
python -u scripts/_rl_train.py \
  --out-dir "$ROOT/runs/hexgt_rl_smoke" \
  --epochs 1 --games-per-epoch 4 --active 4 --vbatch 8 --visits 32 --max-actions 40 \
  --train-steps-per-epoch 5 --batch 16 --warmup 2 \
  --eval-every 1 --eval-games 2 --eval-visits 32 --sealbot --no-compile 2>&1 \
  | grep -vE "UserWarning|frombuffer|writable|suppressed|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
echo "=== rl_train.log ==="
cat "$ROOT/runs/hexgt_rl_smoke/rl_train.log" 2>/dev/null
echo "=== checkpoints ==="
ls -la "$ROOT/runs/hexgt_rl_smoke/checkpoints" 2>/dev/null
echo "=== eval json ==="
cat "$ROOT/runs/hexgt_rl_smoke/eval/epoch_000000_eval.json" 2>/dev/null
