#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/e/Hexo-BotTrainer
export VIRTUAL_ENV=/root/.venvs/hexo-bottrainer-wsl
export PATH="/root/.cargo/bin:$VIRTUAL_ENV/bin:$PATH"
echo "=== cargo: $(command -v cargo) ($(cargo --version)) ==="
echo "=== incremental maturin develop hexo_models (release) ==="
cd "$ROOT/packages/hexo_models"
/root/.venvs/hexo-bottrainer-wsl/bin/python -m maturin develop -m Cargo.toml --features python --release 2>&1 | tail -20
rc=${PIPESTATUS[0]}
echo "=== maturin rc=$rc ==="
[ "$rc" -ne 0 ] && exit "$rc"
bash "$ROOT/scripts/_run_profile.sh"
