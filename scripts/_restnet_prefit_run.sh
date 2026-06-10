#!/usr/bin/env bash
# Launch the dense_cnn_restnet HF prefit (convert human games -> shards -> prefit),
# detached on the GPU. Conversion is CPU (24 workers); prefit is GPU.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
PRE=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit
mkdir -p "$PRE"
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_restnet_prefit.sh > /tmp/_restnet_prefit_clean.sh
setsid bash /tmp/_restnet_prefit_clean.sh \
  --config /mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_restnet_main1.toml \
  --jsonl "$PRE/hf_corpus/hexo_human_corpus.jsonl" \
  --out "$PRE/restnet_hf_prefit.pt" \
  --threads 24 --prefit-epochs 1 --prefit-batch 32 --skip-generation \
  > "$PRE/prefit.log" 2>&1 &
echo "prefit launched pid=$! log=$PRE/prefit.log"
sleep 2
echo "--- early log ---"; tail -5 "$PRE/prefit.log" 2>/dev/null
