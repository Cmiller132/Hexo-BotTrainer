#!/usr/bin/env bash
set -uo pipefail
for i in $(seq 1 25); do
  alive=$(pgrep -af 'supervise_target_64x4_wsl.sh|train_model .*dense_cnn_model1_target_64x4' | grep -v pgrep || true)
  [ -z "$alive" ] && break
  sleep 2
done
alive=$(pgrep -af 'supervise_target_64x4_wsl.sh|train_model .*dense_cnn_model1_target_64x4' | grep -v pgrep || true)
echo "remaining 64x4 procs: ${alive:-none}"
echo "=== python procs (any) ==="
pgrep -af 'venvs/hexo-bottrainer-wsl/bin/python' | grep -v pgrep || echo none
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
