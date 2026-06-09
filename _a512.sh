export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexgnn/python:$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
echo "########## active=512 (opening throughput, contended by live run) ##########"
timeout 90 $VIRTUAL_ENV/bin/python -u $R/_hexgnn_posps.py 96 2 0 512 512 64 512 16 2 4 2>&1 | grep -iE "pos/s|CONFIG" | grep -v "mcts-trace"
