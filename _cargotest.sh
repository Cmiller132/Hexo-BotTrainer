export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYO3_PYTHON="$(which python)"
cargo test -p hexo_models --features python mcts_tree:: 2>&1 | tail -30
echo "EXIT=${PIPESTATUS[0]}"
