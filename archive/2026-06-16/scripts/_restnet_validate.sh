#!/usr/bin/env bash
# CPU-only validation of the HF replay converter on a sample of games.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
J=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit/hf_corpus/hexo_human_corpus.jsonl
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_restnet_prefit.sh > /tmp/_restnet_prefit_clean.sh
bash /tmp/_restnet_prefit_clean.sh \
  --config /mnt/e/Hexo-BotTrainer-hexgt/configs/dense_cnn_restnet_main1.toml \
  --jsonl "$J" --validate 40
