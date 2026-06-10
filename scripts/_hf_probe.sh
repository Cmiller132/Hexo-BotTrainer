#!/usr/bin/env bash
# Probe: can the WSL training env download the HF dataset? (tooling + network)
PY=/root/.venvs/hexgt-build/bin/python
echo "=== python ==="; "$PY" --version 2>&1
echo "=== huggingface_hub ==="; "$PY" -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>&1 | tail -1
echo "=== datasets ==="; "$PY" -c "import datasets; print(datasets.__version__)" 2>&1 | tail -1
echo "=== pyarrow ==="; "$PY" -c "import pyarrow; print(pyarrow.__version__)" 2>&1 | tail -1
echo "=== network: HEAD huggingface.co ==="; curl -sS -m 15 -I https://huggingface.co 2>&1 | head -3
echo "=== network: HEAD dataset resolve root ==="; curl -sS -m 15 -I "https://huggingface.co/api/datasets/timmyburn/hexo-bootstrap-corpus" 2>&1 | head -3
echo "=== git-lfs ==="; which git-lfs 2>&1; git lfs version 2>&1 | head -1
