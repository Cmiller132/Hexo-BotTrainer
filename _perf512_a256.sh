#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/_perf_512.py \
  --shards runs/hexgt_rl_main/selfplay \
  --visits 512 --active 256 --vbatch 64 --moves 2 \
  --compile \
  2>&1 | tee runs/_perf_512_a256.log
echo "EXIT=${PIPESTATUS[0]}"
