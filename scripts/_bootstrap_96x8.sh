#!/usr/bin/env bash
# Generate the SealBot-bootstrapped prefit for the 96x8+P7 architecture (its shape
# differs from 96x6 so the 96x6 prefit can't be reused). Plays SealBot(best)-vs-
# SealBot games and prefits the dense net so the run starts from a sensible policy
# instead of ~10 epochs of zeros. Mirrors _bootstrap_64x4.sh.
set -uo pipefail
source /root/.venvs/hexo-bottrainer-wsl/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer
SEALBOT=/mnt/e/SealBot
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python:$SEALBOT:$SEALBOT/best"
export SEALBOT_PATH="$SEALBOT"
export OMP_NUM_THREADS=8
cd "$ROOT"
mkdir -p runs/dense_cnn_model1_target_96x8/checkpoints
python scripts/bootstrap_dense_cnn_sealbot.py \
  --config configs/dense_cnn_model1_target_96x8.toml \
  --out runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt \
  --sealbot-path "$SEALBOT" \
  --target-samples 5000 --threads 32 --turn-time-ms 100 --prefit-epochs 8 2>&1 | tail -40
echo "=== bootstrap rc=${PIPESTATUS[0]} ==="
ls -la runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt 2>/dev/null
