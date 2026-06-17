#!/usr/bin/env bash
# Download the HF human corpus JSONL into a scratch dir and peek at it (read-only).
set -uo pipefail
DST=/mnt/e/Hexo-BotTrainer-hexgt/runs/dense_cnn_restnet_main1_prefit/hf_corpus
mkdir -p "$DST"
URL="https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus/resolve/main/hexo_human_corpus.jsonl"
echo "downloading $URL"
curl -sS -L -m 120 -o "$DST/hexo_human_corpus.jsonl" "$URL"
echo "=== size ==="; ls -la "$DST/hexo_human_corpus.jsonl"
echo "=== line count ==="; wc -l < "$DST/hexo_human_corpus.jsonl"
echo "=== first game (line 1) ==="; head -1 "$DST/hexo_human_corpus.jsonl"
echo
echo "=== SCHEMA.md ==="
curl -sS -L -m 60 "https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus/resolve/main/SCHEMA.md" 2>/dev/null | head -60
