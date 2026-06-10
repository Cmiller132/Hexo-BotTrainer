#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
# rustup's toolchain (cargo 1.95, lockfile v4) — NOT the older apt cargo a
# non-login shell otherwise picks up (fails: "lock file version 4 requires
# -Znext-lockfile-bump").
export PATH="/root/.cargo/bin:$PATH"

python - <<'PY'
import hexo_models
try:
    from hexo_models import _rust
except Exception as exc:
    _rust = exc
print("hexo_models:", getattr(hexo_models, "__file__", None))
print("hexo_models._rust:", getattr(_rust, "__file__", _rust))
PY

# --release is mandatory: a debug-profile MCTS/encoding crate is ~10x slower
# and would silently tank self-play throughput.
maturin develop --release -m packages/hexo_models/Cargo.toml --features python
