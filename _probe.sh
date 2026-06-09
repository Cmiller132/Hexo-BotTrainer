#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
SHARDS=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay
echo "===== vbatch=16 eager ====="
python -u scripts/_mcts_selfplay_probe.py --shards "$SHARDS" --games 64 --visits 128 --moves 6 --vbatch 16 2>&1 | tail -20
