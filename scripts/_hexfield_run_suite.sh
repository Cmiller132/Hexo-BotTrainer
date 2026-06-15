#!/usr/bin/env bash
# Full hexfield test suite, CPU-only (the GPU is owned by the live run; the
# fp16-CUDA test auto-skips). Run in the authoritative hexgt-build venv via the
# testkit sys.path shim. Verifies the audit-fix application.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_hexfield_geometry.py \
  tests/test_hexfield_support.py \
  tests/test_hexfield_features.py \
  tests/test_hexfield_model.py \
  tests/test_hexfield_targets.py \
  tests/test_hexfield_rust_parity.py \
  tests/test_hexfield_search_parity.py \
  tests/test_hexfield_continuous_parity.py \
  tests/test_hexfield_seed_contract.py \
  tests/test_hexfield_divergence_properties.py \
  tests/test_hexfield_plugin.py \
  -q --no-header -p no:cacheprovider 2>&1 | tail -60
echo "PYTEST_EXIT=${PIPESTATUS[0]}"
