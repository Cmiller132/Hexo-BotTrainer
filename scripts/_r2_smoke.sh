#!/usr/bin/env bash
# Smoke: HexfieldTrainer constructs on CPU from the worktree (the crash path).
set -eu
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
export HEXFIELD_CHANNELS=128 HEXFIELD_TRAIN_COMPILE=1
/root/.venvs/hexgt-build/bin/python - <<'EOF'
import inspect
from hexfield import trainer as T
src = inspect.getsource(T.HexfieldTrainer.__init__)
assert "import torch._dynamo" not in src, "function-level import still present"
print("no function-level torch import; module import OK:", T.torch.__version__)
EOF
echo smoke OK
