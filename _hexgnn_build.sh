#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
export VIRTUAL_ENV=/root/.venvs/hexgt-build
cd /mnt/e/Hexo-BotTrainer-hexgt
/root/.venvs/hexgt-build/bin/maturin develop -m packages/hexo_models/Cargo.toml --features python --release
