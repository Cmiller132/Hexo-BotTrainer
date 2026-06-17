#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
P=/root/.venvs/hexgt-build/bin/python
$P _wf_s3_post.py _wf_s3_full_out.json _wf_s3_post_full.txt > _wf_s3_post_full_run.txt 2>&1
echo "post exit=$?" >> _wf_s3_post_full_run.txt
$P _wf_s3_post3.py _wf_s3_full_out.json _wf_s3_post3_out.txt > _wf_s3_post3_run.txt 2>&1
echo "post3 exit=$?" >> _wf_s3_post3_run.txt
