#!/bin/bash
# Reference-LADDER extension: the canonical 48-test pipeline baseline, BEFORE edits.
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python -m pytest \
  tests/test_dense_cnn_pipeline.py tests/test_dense_cnn_restnet.py \
  -q --no-header -p no:cacheprovider \
  > scripts/_wf_evref2_pipeline_before.txt 2>&1
echo "exit=$?" >> scripts/_wf_evref2_pipeline_before.txt
