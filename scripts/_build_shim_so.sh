#!/usr/bin/env bash
# Isolated release build of the hexfield native module + a PYTHONPATH shim dir
# (/tmp/hexshim/hexfield -> dev-tree python package with the fresh .so).
# Does NOT touch the live gumbel worktree.
set -eu
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
cd /mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield
export CARGO_TARGET_DIR=/tmp/hexbuild
/root/.venvs/hexfield-dev/bin/maturin build --release -o /tmp/hexwheels
WHEEL=$(ls -t /tmp/hexwheels/hexfield-*.whl | head -1)
rm -rf /tmp/hexshim && mkdir -p /tmp/hexshim
cd /tmp/hexshim
/root/.venvs/hexfield-dev/bin/python -m zipfile -e "$WHEEL" wheel_contents
# Shim layout: copy the DEV python package, then drop in the fresh .so.
cp -r /mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python/hexfield /tmp/hexshim/hexfield
SO=$(find wheel_contents -name '_rust*.so' | head -1)
cp "$SO" /tmp/hexshim/hexfield/
echo "shim ready: /tmp/hexshim  ($(basename "$SO") $(stat -c%s "$SO") bytes)"
