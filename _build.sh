#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
echo "=== cargo: $(which cargo) ==="
cargo --version
echo "=== maturin develop --release (hexo_models, --features python) ==="
maturin develop --release -m packages/hexo_models/Cargo.toml --features python 2>&1 | tail -45
echo "EXIT=$?"
