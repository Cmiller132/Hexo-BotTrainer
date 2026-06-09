#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -m pytest tests/test_hexgt_parallel_eval.py -q 2>&1 | tail -25
echo "EXIT=${PIPESTATUS[0]}"
