export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
R=/mnt/e/Hexo-BotTrainer-hexgt
export SEALBOT_PATH=/mnt/e/SealBot
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python:$R/packages/hexo_frontend/python:$SEALBOT_PATH:$SEALBOT_PATH/best"
cd "$R"
OUTDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1_prefit
mkdir -p "$OUTDIR"
BLOG="$OUTDIR/bootstrap.log"
echo "=== bootstrap start $(date '+%H:%M:%S') (64x10, target-samples=5000, prefit-epochs=4, turn=100ms) ===" > "$BLOG"
setsid "$VIRTUAL_ENV/bin/python" -u scripts/bootstrap_dense_cnn_sealbot.py \
  --config "$R/configs/dense_cnn_rl_main1.toml" \
  --out "$OUTDIR/bootstrap_sealbot_prefit.pt" \
  --sealbot-path "$SEALBOT_PATH" --sealbot-variant best \
  --target-samples 5000 --threads 32 --turn-time-ms 100 --prefit-epochs 4 --seed 1 \
  >> "$BLOG" 2>&1 &
echo "bootstrap setsid pid=$!  log=$BLOG"
sleep 3
echo "--- early log ---"; tail -5 "$BLOG"
