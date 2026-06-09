#!/usr/bin/env bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/_sweep_active.py --shards runs/hexgt_rl_main/selfplay --moves 4 --repeat 2 2>&1 \
  | grep -vE "UserWarning|frombuffer|W0602|torch/_dynamo|Graph break|capture_scalar|max_autotune|^\s*$"
echo "EXIT=${PIPESTATUS[0]}"
