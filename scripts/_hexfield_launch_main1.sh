#!/usr/bin/env bash
# Launch the hexfield production run (hexfield_main_1.toml) as a detached
# daemon (setsid) so it survives this shell. 200 epochs, 512 visits, PCR
# fast=128 @ 33%, warm-started from the BC prefit. Output to the dashboard
# mount (/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1). CPU/GPU: full GPU.
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="packages/hexfield/python:packages/dense_cnn_restnet/python"
export OMP_NUM_THREADS=8
TS=$(date +%Y%m%d_%H%M%S)
LOG="runs/_hexfield_main1_launch_${TS}.log"
mkdir -p runs
setsid nohup /root/.venvs/hexgt-build/bin/python -u \
  -m hexo_train.cli.train_model configs/hexfield_main_1.toml \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "LAUNCHED pid=$PID log=$LOG"
echo "$PID" > runs/_hexfield_main1.pid
sleep 2
echo "--- alive? ---"
ps -o pid,etime,cmd -p "$PID" 2>/dev/null | tail -2
