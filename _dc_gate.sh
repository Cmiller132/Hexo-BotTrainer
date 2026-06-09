export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12
R=/mnt/e/Hexo-BotTrainer-hexgt
export SEALBOT_PATH=/mnt/e/SealBot
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python:$R/packages/hexo_frontend/python:$SEALBOT_PATH:$SEALBOT_PATH/best"
cd "$R"
mkdir -p /mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
GLOG=/mnt/e/Hexo-BotTrainer-hexgt/_dc_gate.out
echo "=== GATE launch (direct CLI, from-scratch) at $(date '+%H:%M:%S') ===" > "$GLOG"
nohup "$VIRTUAL_ENV/bin/python" -u -m hexo_train.cli.train_model "$R/configs/dense_cnn_rl_main1.toml" >> "$GLOG" 2>&1 &
echo "gate pid=$!"
