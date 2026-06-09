#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python -m pytest tests/test_hexgt_forward_chunk.py tests/test_hexgt_inference.py tests/test_hexgt_mcts.py tests/test_hexgt_feature_buffer.py -q 2>&1 | tail -20
echo "EXIT=${PIPESTATUS[0]}"
