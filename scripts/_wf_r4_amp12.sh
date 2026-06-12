#!/bin/bash
# ckpt12 amplitude on the SAME ep10 long games as the ckpt10/ckpt6 baselines.
cd /mnt/e/Hexo-BotTrainer-hexgt/scripts || exit 1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=6
PY=/root/.venvs/hexgt-build/bin/python
P=archive/_wf_r4_amp_probe.py
if [ ! -f "$P" ]; then P=_wf_r4_amp_probe.py; fi
echo "using probe: $P" > _wf_r4_amp12_run.log
$PY "$P" 12 10 8 4 _wf_r4_amp_c12_g10.json >> _wf_r4_amp12_run.log 2>&1
echo "=== c12_g10 exit=$? ===" >> _wf_r4_amp12_run.log
echo DONE > _wf_r4_amp12_done.txt
