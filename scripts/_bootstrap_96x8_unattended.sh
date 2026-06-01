#!/usr/bin/env bash
# Unattended 96x8 SealBot bootstrap: runs the generator+prefit directly (FULL
# unbuffered output to a log so progress is monitorable live), writes an rc
# sentinel on exit. Launch this under setsid nohup. 5000 samples @ 100ms + prefit.
set -uo pipefail
source /root/.venvs/hexo-bottrainer-wsl/bin/activate
ROOT=/mnt/e/Hexo-BotTrainer
SEALBOT=/mnt/e/SealBot
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python:$SEALBOT:$SEALBOT/best"
export SEALBOT_PATH="$SEALBOT"
export OMP_NUM_THREADS=8
cd "$ROOT"
DIAG="$ROOT/runs/dense_cnn_model1_target_96x8/diagnostics"
LOG="$DIAG/bootstrap.log"
RC="$DIAG/bootstrap.rc"
mkdir -p "$DIAG" "$ROOT/runs/dense_cnn_model1_target_96x8/checkpoints"
rm -f "$RC"
echo "=== bootstrap start $(date -Is) ===" > "$LOG"
python -u scripts/bootstrap_dense_cnn_sealbot.py \
  --config configs/dense_cnn_model1_target_96x8.toml \
  --out runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt \
  --sealbot-path "$SEALBOT" \
  --target-samples 5000 --threads 32 --turn-time-ms 100 --prefit-epochs 8 >>"$LOG" 2>&1
rc=$?
echo "=== bootstrap end $(date -Is) rc=$rc ===" >> "$LOG"
echo "$rc" > "$RC"
