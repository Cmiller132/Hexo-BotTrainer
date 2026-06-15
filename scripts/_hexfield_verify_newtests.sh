#!/usr/bin/env bash
# Verbose run of just the new audit-fix tests to confirm each executed (not
# silently skipped), and surface skip reasons for the 2 suite skips.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_hexfield_model.py::test_bias_gather_backward_equals_generic_indexing \
  tests/test_hexfield_model.py::test_fresh_model_zero_init_residual_identity \
  tests/test_hexfield_model.py::test_sdpa_equals_materialized_fp16_cuda \
  tests/test_hexfield_targets.py::test_legacy_shard_schema_version_guard \
  tests/test_hexfield_plugin.py::test_config_rejects_unknown_top_level_keys \
  tests/test_hexfield_support.py \
  -v -rs --no-header -p no:cacheprovider 2>&1 | tail -40
echo "PYTEST_EXIT=${PIPESTATUS[0]}"
