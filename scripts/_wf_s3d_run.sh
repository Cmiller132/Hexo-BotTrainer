#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
exec /root/.venvs/hexgt-build/bin/python _wf_s3d_probe.py > _wf_s3d_probe_run.txt 2>&1
