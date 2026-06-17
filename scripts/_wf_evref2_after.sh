#!/bin/bash
# Reference-LADDER extension: AFTER-edit verification.
# 1. compile check, 2. reference-eval suite, 3. the 48-test pipeline baseline.
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
PY=/root/.venvs/hexgt-build/bin/python
{
  echo "== compile check =="
  $PY -m py_compile \
    packages/dense_cnn_restnet/python/dense_cnn_restnet/config.py \
    packages/dense_cnn_restnet/python/dense_cnn_restnet/evaluation.py \
    tests/test_restnet_reference_eval.py && echo "compile ok"
  echo "== reference-eval suite =="
  $PY -m pytest tests/test_restnet_reference_eval.py -q --no-header -p no:cacheprovider
  echo "ref_exit=$?"
  echo "== 48-test pipeline baseline =="
  $PY -m pytest tests/test_dense_cnn_pipeline.py tests/test_dense_cnn_restnet.py -q --no-header -p no:cacheprovider
  echo "pipeline_exit=$?"
} > scripts/_wf_evref2_after.txt 2>&1
echo "exit=$?" >> scripts/_wf_evref2_after.txt
