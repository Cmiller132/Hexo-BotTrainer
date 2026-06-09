#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -u _vram_profile2.py 2>&1 | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
echo "EXIT=${PIPESTATUS[0]}"
