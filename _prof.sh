export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export HEXO_MCTS_TRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexgnn/python:$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
# td gnn pcr visits active num_games max_actions vbatch N STEER
timeout 170 $VIRTUAL_ENV/bin/python -u $R/_hexgnn_posps.py 96 2 0 512 256 96 512 16 2 4 \
  > $R/_prof.out 2>&1
echo "exit=$?"
