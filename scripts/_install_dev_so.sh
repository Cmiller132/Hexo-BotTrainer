#!/usr/bin/env bash
# Install the freshly-built shim .so into the DEV tree package and smoke-import.
set -eu
SRC=/tmp/hexshim/hexfield/_rust.cpython-312-x86_64-linux-gnu.so
DST=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python/hexfield/_rust.cpython-312-x86_64-linux-gnu.so
md5sum "$SRC" "$DST" || true
cp "$SRC" "$DST"
md5sum "$DST"
PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python \
  /root/.venvs/hexgt-build/bin/python -c 'import hexfield._rust as r; print("import OK:", r.__file__)'
