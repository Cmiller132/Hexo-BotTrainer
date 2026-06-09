#!/usr/bin/env bash
# Rebuild the STANDALONE hexo_engine .so in the -hexgt worktree (the one the live
# run's PYTHONPATH and the tests' sys.path.insert actually load) so its
# RustHexoState layout matches the engine crate embedded in hexo_models.
set -uo pipefail
source /root/.venvs/hexgt-build/bin/activate
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "maturin: $(which maturin)"
echo "=== maturin develop --release hexo_engine (from -hexgt) ==="
maturin develop --release -m packages/hexo_engine/Cargo.toml --features python 2>&1 | tail -8
echo "=== rc=${PIPESTATUS[0]} ==="
ls -la packages/hexo_engine/python/hexo_engine/_rust*.so
