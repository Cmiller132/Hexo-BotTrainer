#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
SHARDS=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay
python -u scripts/_bc_train.py --shards "$SHARDS" \
  --train-epochs "${TRAIN_EPOCHS:-20,21,23,24}" --shards-per-epoch "${SPE:-120}" \
  --heldout-shards "${HOS:-60}" --target-steps "${STEPS:-15000}" \
  --group-size "${GROUP:-40}" --ckpt-every "${CKPT:-1500}" --eval-every "${EVAL:-1500}" \
  --lr "${LR:-0.001}" --warmup "${WARMUP:-500}" --out-dir "${OUT:-/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc}" \
  2>&1 | tail -60
