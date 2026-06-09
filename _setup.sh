#!/bin/bash
# Repoint the ISOLATED dev venv (hexgt-build, NOT the live venv) at the -hexgt
# worktree and rebuild its native module THERE. Never touches the live tree.
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
V=/root/.venvs/hexgt-build/bin
if [ ! -x "$V/python" ]; then echo "venv missing — run full setup"; exit 1; fi
for p in hexo_engine hexo_utils hexo_runner hexo_train hexo_models hexo_frontend; do
  echo "=== install $p (editable, against -hexgt) ==="
  "$V/pip" install -e "./packages/$p" -q || { echo "FAILED $p"; exit 1; }
done
echo "=== verify points at -hexgt + native submods ==="
"$V/pip" show -f hexo-models 2>/dev/null | grep -E "Editable|Location" | head -2
"$V/python" -c "import hexo_models._rust as r; print('rust submods:', [s for s in dir(r) if not s.startswith(chr(95))]); from hexo_models.hexgt import rust_bridge; print('caps:', dict(rust_bridge.capabilities()))"
echo "SETUP_OK"
