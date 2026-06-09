#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
for B in 1 2 4 8 32; do
  python -u scripts/_eval_pathbench.py --batch $B --iters 60 2>&1 | grep -vE "UserWarning|node_feat = torch"
  echo "---"
done
