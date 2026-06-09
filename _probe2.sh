#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
SHARDS=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay
for VB in 32 64; do
  echo "===== vbatch=$VB eager ====="
  python -u scripts/_mcts_selfplay_probe.py --shards "$SHARDS" --games 64 --visits 128 --moves 6 --vbatch $VB 2>&1 | grep -E "searched positions/sec|forwards/searched|time split|cache hit"
done
echo "===== vbatch=64 COMPILE ====="
python -u scripts/_mcts_selfplay_probe.py --shards "$SHARDS" --games 64 --visits 128 --moves 6 --vbatch 64 --compile 2>&1 | grep -E "searched positions/sec|forwards/searched|time split|cache hit"
