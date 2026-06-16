#!/usr/bin/env bash
set -uo pipefail
source /root/.venvs/hexo-bottrainer-wsl/bin/activate
# rustup's toolchain (cargo 1.95, lockfile v4) — NOT the older apt cargo that a
# non-login shell would otherwise pick up.
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer
echo "cargo: $(which cargo) $(cargo --version)"
echo "=== maturin develop --release (hexo_models) ==="
maturin develop --release -m packages/hexo_models/Cargo.toml --features python 2>&1 | tail -30
echo "=== rc=${PIPESTATUS[0]} ==="
