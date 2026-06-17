#!/bin/bash
# Regression baseline for the reference-eval feature (run BEFORE the edits).
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_pipeline.py tests/test_dense_cnn_restnet.py \
  -q --no-header -p no:cacheprovider \
  > scripts/_wf_evref_baseline.txt 2>&1
echo "exit=$?" >> scripts/_wf_evref_baseline.txt
