export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexgnn/python:$R/packages/hexo_engine/python:$R/packages/hexo_models/python"
# opening density at one full chunk (1024 leaves), then a midgame-ish density
timeout 300 $VIRTUAL_ENV/bin/python -u $R/_fwdbench.py 1024 222 1499 215 2>&1 | grep -vE "UserWarning|frombuffer|index_reduce|beta"
echo "--- midgame density (smaller G to fit) ---"
timeout 300 $VIRTUAL_ENV/bin/python -u $R/_fwdbench.py 512 600 4500 400 2>&1 | grep -vE "UserWarning|frombuffer|index_reduce|beta"
