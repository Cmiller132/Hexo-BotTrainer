#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
P=/root/.venvs/hexgt-build/bin/python
$P _wf_s3_mlaudit.py 2 _wf_s3_smoke_out.json > _wf_s3_smoke_run.txt 2>&1
echo "exit=$?" >> _wf_s3_smoke_run.txt
