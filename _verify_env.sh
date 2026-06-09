#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
PY=/root/.venvs/hexgt-build/bin/python
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
echo "=== which python ==="
echo "$PY"
"$PY" -c "import sys; print('exe', sys.executable)"
echo "=== torch / cuda ==="
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
EOF
echo "=== hexo_models editable location ==="
"$PY" -c "import hexo_models, hexo_models.hexgt as h; print('hexo_models at', hexo_models.__file__)"
echo "=== rust submods + caps ==="
"$PY" -c "import hexo_models._rust as r; print('rust submods:', [s for s in dir(r) if not s.startswith(chr(95))]); from hexo_models.hexgt import rust_bridge; print('caps:', dict(rust_bridge.capabilities()))"
echo "VERIFY_OK"
