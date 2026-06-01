#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexo-bottrainer-wsl/bin/activate
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer
echo "cargo: $(which cargo) $(cargo --version)"
echo "=== FULL cargo clean (pyo3/rayon rlibs were built by an older rustc -> E0461 ICE) ==="
cargo clean 2>&1 | tail -5
echo "=== maturin develop --release (hexo_models) ==="
maturin develop --release -m packages/hexo_models/Cargo.toml --features python 2>&1 | tail -20
echo "=== rc=${PIPESTATUS[0]} ==="
