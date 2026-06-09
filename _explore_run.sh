#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -u _explore_readonly.py /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main 150 2>&1 | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
