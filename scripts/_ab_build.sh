#!/usr/bin/env bash
# Build hexo_models (with the embedded hexo_engine crate change) into the LIVE
# hexgt-build venv, from THIS worktree. This replaces the in-place _rust .so the
# main3 run uses — only run while the run is stopped.
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "venv: $VIRTUAL_ENV"
echo "cargo: $(which cargo) $(cargo --version)"
echo "=== maturin develop --release -m packages/hexo_models/Cargo.toml --features python ==="
maturin develop --release -m packages/hexo_models/Cargo.toml --features python 2>&1 | tail -25
echo "=== rc=${PIPESTATUS[0]} ==="
