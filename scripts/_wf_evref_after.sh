#!/bin/bash
# Post-change regression: baseline suite + restnet config consumers + new tests.
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_pipeline.py \
  tests/test_dense_cnn_restnet.py \
  tests/test_restnet_reference_eval.py \
  tests/test_restnet_frozen_win_smoke.py \
  tests/test_restnet_length_decay.py \
  tests/test_dense_cnn_restnet_pcr_policy_init.py \
  tests/test_dense_cnn_continuous_scheduler.py \
  -q --no-header -p no:cacheprovider \
  > scripts/_wf_evref_after.txt 2>&1
echo "exit=$?" >> scripts/_wf_evref_after.txt
