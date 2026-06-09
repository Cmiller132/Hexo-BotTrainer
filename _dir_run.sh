#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export CUDA_VISIBLE_DEVICES=""   # force CPU; don't touch the live GPU run
OMP_NUM_THREADS=4 python -u _explore_dirichlet.py 2>&1 | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
