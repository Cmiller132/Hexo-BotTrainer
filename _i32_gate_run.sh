#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$PWD/packages/hexgnn/python:$PWD/packages/hexo_engine/python:$PWD/packages/hexo_utils/python:$PWD/packages/hexo_runner/python:$PWD/packages/hexo_train/python:$PWD/packages/hexo_models/python"
echo "=== BIT-IDENTITY GATE ==="
python _i32_gate.py
echo "=== PARITY TEST ==="
python -m pytest tests/test_hexgnn_featurizer_parity.py -q 2>&1 | tail -15
