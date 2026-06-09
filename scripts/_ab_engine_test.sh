#!/usr/bin/env bash
set -uo pipefail
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "cargo: $(cargo --version)"
echo "=== cargo test -p hexo_engine (tactics + state: live_threats invariant) ==="
cargo test -p hexo_engine --lib 2>&1 | tail -40
echo "=== rc=${PIPESTATUS[0]} ==="
echo "=== cargo check -p hexo_models --features python (hexgt threats.rs short-circuit) ==="
cargo check -p hexo_models --features python 2>&1 | tail -25
echo "=== rc=${PIPESTATUS[0]} ==="
