#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
echo "############ default (RAYON_NUM_THREADS unset) ############"
python -u scripts/_feat_probe.py --shards runs/hexgt_rl_main/selfplay --states 1024 --iters 20
echo ""
echo "############ RAYON_NUM_THREADS=1 (serial baseline) ############"
RAYON_NUM_THREADS=1 python -u scripts/_feat_probe.py --shards runs/hexgt_rl_main/selfplay --states 1024 --iters 20
echo "EXIT=$?"
