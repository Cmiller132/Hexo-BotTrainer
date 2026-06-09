#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -m pytest tests/test_hexgt_selfplay.py tests/test_hexgt_model.py tests/test_hexgt_inference.py tests/test_hexgt_contract.py -q 2>&1 | tail -15
echo "EXIT=${PIPESTATUS[0]}"
