#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python _sp_smoke.py --games 24 --active 24 --vbatch 64 --visits 128 --max-actions 200 --compile 2>&1 | grep -v "UserWarning\|frombuffer\|writable\|suppressed"
echo "EXIT=${PIPESTATUS[0]}"
