#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "=== maturin develop --release (hexo_models, includes hexgnn crate) ==="
maturin develop --release -m packages/hexo_models/Cargo.toml --features python 2>&1 | tail -25
echo "=== BUILD DONE ==="
