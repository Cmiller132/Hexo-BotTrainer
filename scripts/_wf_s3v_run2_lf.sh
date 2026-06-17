#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
/root/.venvs/hexgt-build/bin/python _wf_s3v_freshaudit.py > _wf_s3v_freshaudit_out.txt 2>&1
echo "exit=$?" >> _wf_s3v_freshaudit_out.txt
