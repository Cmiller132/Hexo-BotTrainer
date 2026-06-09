#!/usr/bin/env bash
# Mirror the tactics.rs WindowStore change into the -hexgtfeat worktree (which the
# STANDALONE hexo_engine extension is editable-installed from) and rebuild that
# .so, so its RustHexoState layout matches the engine crate embedded in
# hexo_models (built from -hexgt). The capsule clone in hexgt/state.rs casts a
# raw handle between the two, so their layouts MUST match.
set -uo pipefail
SRC=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/rust/src/tactics.rs
DST=/mnt/e/Hexo-BotTrainer-hexgtfeat/packages/hexo_engine/rust/src/tactics.rs
echo "=== copy tactics.rs -hexgt -> -hexgtfeat ==="
cp "$SRC" "$DST"
diff "$SRC" "$DST" && echo "(identical after copy)"

source /root/.venvs/hexgt-build/bin/activate
export PATH="/root/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgtfeat
echo "=== maturin develop --release hexo_engine (from -hexgtfeat) ==="
maturin develop --release -m packages/hexo_engine/Cargo.toml --features python 2>&1 | tail -15
echo "=== rc=${PIPESTATUS[0]} ==="
ls -la packages/hexo_engine/python/hexo_engine/_rust*.so
