#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  python -u scripts/_perf_512.py --shards runs/hexgt_rl_main/selfplay \
    --visits 512 --active 64 --vbatch 64 --moves 3 --compile 2>&1 | grep -E 'THROUGHPUT'
}
for i in 1 2 3; do
  echo -n "PIN=on  "; HEXGT_PIN_HOST=1 run
done
for i in 1 2 3; do
  echo -n "PIN=off "; HEXGT_PIN_HOST=0 run
done
