#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
P=/root/.venvs/hexgt-build/bin/python
setsid nohup $P _wf_s3_mlaudit.py 60 _wf_s3_full_out.json > _wf_s3_full_run.txt 2>&1 < /dev/null &
echo "launched pid=$!"
