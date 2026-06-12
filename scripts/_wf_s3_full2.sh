#!/bin/bash
setsid bash -c 'cd /mnt/e/Hexo-BotTrainer-hexgt/scripts && export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 && exec /root/.venvs/hexgt-build/bin/python _wf_s3_mlaudit.py 60 _wf_s3_full_out.json' < /dev/null > /mnt/e/Hexo-BotTrainer-hexgt/scripts/_wf_s3_full_run.txt 2>&1 &
echo "launched setsid child"
