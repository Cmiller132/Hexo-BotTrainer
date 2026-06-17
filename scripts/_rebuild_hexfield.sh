#!/usr/bin/env bash
set -euo pipefail

# hexfield Rust rebuild — ALWAYS the isolated hexfield-dev venv, never the
# live hexgt-build venv (the live supervisor relaunches processes between
# epochs and would pick up a replaced .so from disk; hexfield ships its own
# cdylib precisely so its builds can never touch the live lineages).

cd /mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexfield-dev/bin/activate
# rustup toolchain (lockfile v4), not the older apt cargo.
export PATH="/root/.cargo/bin:$PATH"

# --release is mandatory: a debug featurizer/search crate is ~10x slower.
maturin develop --release -m packages/hexfield/Cargo.toml

# maturin develop installs into the hexfield-dev venv; the parity tests run
# in the hexgt-build venv via PYTHONPATH shims and import the package from
# the source tree, so mirror the built extension into the tree.
SO=$(ls /root/.venvs/hexfield-dev/lib/python3.12/site-packages/hexfield/_rust*.so 2>/dev/null | head -1)
if [ -n "${SO:-}" ]; then
  cp "$SO" packages/hexfield/python/hexfield/
  echo "copied $(basename "$SO") into packages/hexfield/python/hexfield/"
fi
ls -la packages/hexfield/python/hexfield/_rust*.so
