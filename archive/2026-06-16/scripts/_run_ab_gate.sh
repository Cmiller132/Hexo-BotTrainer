#!/usr/bin/env bash
# A/B gate: lockstep vs continuous on the live checkpoint (GPU, real positions).
set -uo pipefail
cd /mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
{
  python scripts/_continuous_ab_gate.py \
    --checkpoint /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1/checkpoints/epoch_000025.pt \
    --positions 256 2>&1 | tail -40
  echo "=== ab gate rc=${PIPESTATUS[0]} ==="
} > _ab_gate_out.txt 2>&1
