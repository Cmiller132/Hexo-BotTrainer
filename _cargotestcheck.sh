export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYO3_PYTHON="$(which python)"
cargo test -p hexo_models --features python --no-run 2>&1 | grep -E "^error|error\[|RustSearch::new|cannot find|expected|previous error" | head -40
echo "EXIT=${PIPESTATUS[0]}"
