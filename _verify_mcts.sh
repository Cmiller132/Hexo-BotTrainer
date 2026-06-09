#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
PY=/root/.venvs/hexgt-build/bin/python
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
"$PY" - <<'EOF'
import hexo_models._rust as r
from hexo_models.hexgt import rust_bridge
print("caps:", dict(rust_bridge.capabilities()))
sess = rust_bridge.new_mcts_session(max_states=4096, n=3)
print("session:", type(sess).__name__, "len", sess.len())
from hexo_models.hexgt import mcts as hmcts
print("python session:", hmcts.new_mcts_session(n=3).__class__.__name__, "OK")
EOF
echo "IMPORT_EXIT=$?"
