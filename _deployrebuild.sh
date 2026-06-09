#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
source /root/.venvs/hexgt-build/bin/activate
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "=== maturin develop (LIVE worktree, features) ==="
maturin develop -m packages/hexo_models/Cargo.toml --features python --release 2>&1 | tail -5
echo "maturin_exit=${PIPESTATUS[0]}"
echo "=== live .so mtime (should be NOW) ==="
ls -la --time-style=+%H:%M packages/hexo_models/python/hexo_models/_rust*.so
