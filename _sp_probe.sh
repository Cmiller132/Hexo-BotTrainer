#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
echo "########## active=64 vbatch=64 COMPILE ##########"
python -u _sp_smoke.py --games 64 --active 64 --vbatch 64 --visits 128 --max-actions 60 --compile 2>/dev/null
echo "EXIT_C=$?"
