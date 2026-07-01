#!/usr/bin/env bash
set -euo pipefail

# hexfield Rust rebuild — GUMBEL WORKTREE variant.
# Worktree-pointed copy of scripts/_rebuild_hexfield.sh: builds the main_6 Gumbel
# branch (claude/hexfield-gumbel) from the ISOLATED worktree tree
# /mnt/e/Hexo-BotTrainer-gumbel using the ISOLATED hexfield-dev venv, never the
# live hexgt-build venv or the live main tree (E:/Hexo-BotTrainer-hexgt). The live
# main_5 supervisor relaunches processes between epochs and would pick up a
# replaced .so from disk; hexfield ships its own cdylib precisely so its builds
# can never touch the live lineages. This script must NOT be run against the live
# tree or venv.

cd /mnt/e/Hexo-BotTrainer-gumbel
source /root/.venvs/hexfield-dev/bin/activate
# rustup toolchain (lockfile v4), not the older apt cargo.
export PATH="/root/.cargo/bin:$PATH"

# --release is mandatory: a debug featurizer/search crate is ~10x slower.
maturin develop --release -m packages/hexfield/Cargo.toml

# maturin develop installs into the hexfield-dev venv; the parity/gumbel tests run
# in the hexgt-build venv via PYTHONPATH shims and import the package from the
# source tree, so mirror the built extension into the WORKTREE source tree.
SO=$(ls /root/.venvs/hexfield-dev/lib/python3.12/site-packages/hexfield/_rust*.so 2>/dev/null | head -1)
if [ -n "${SO:-}" ]; then
  cp "$SO" packages/hexfield/python/hexfield/
  echo "copied $(basename "$SO") into packages/hexfield/python/hexfield/"
fi
ls -la packages/hexfield/python/hexfield/_rust*.so
