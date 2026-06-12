cd /mnt/e/Hexo-BotTrainer-hexgt/scripts
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4
setsid /root/.venvs/hexgt-build/bin/python _wf_s3d_probe.py > _wf_s3d_probe_run.txt 2>&1 < /dev/null &
echo LAUNCHED
