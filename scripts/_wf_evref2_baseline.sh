#!/bin/bash
# Reference-LADDER extension: BEFORE-edit baseline (single-anchor feature as landed).
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_pipeline.py tests/test_restnet_reference_eval.py \
  -q --no-header -p no:cacheprovider \
  > scripts/_wf_evref2_baseline.txt 2>&1
echo "exit=$?" >> scripts/_wf_evref2_baseline.txt
