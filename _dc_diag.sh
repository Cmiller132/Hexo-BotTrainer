export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export SEALBOT_PATH=/mnt/e/SealBot
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python:$R/packages/hexo_frontend/python:$SEALBOT_PATH:$SEALBOT_PATH/best"
cd "$R"
echo "=== run CLI foreground (capture error) ==="
timeout 120 "$VIRTUAL_ENV/bin/python" -u -m hexo_train.cli.train_model "$R/configs/dense_cnn_rl_main1.toml" 2>&1 | head -40
echo "exit=${PIPESTATUS[0]}"
