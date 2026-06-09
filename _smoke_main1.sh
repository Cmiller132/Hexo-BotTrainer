export PATH="$HOME/.cargo/bin:$PATH"
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
PP=""
for p in hexo_engine hexo_utils hexo_runner hexo_train hexo_models; do PP="$PP:$ROOT/packages/$p/python"; done
PP="$PP:$ROOT/packages/hexgnn/python"
export PYTHONPATH="${PP#:}"
rm -rf "$ROOT/runs/hexgnn_smoke_main1"
/root/.venvs/hexgt-build/bin/python -u "$ROOT/scripts/_rl_train_hexgnn.py" \
  --from-scratch --token-dim 96 --gnn-layers 2 --steerable-channels 4 --n 2 \
  --out-dir "$ROOT/runs/hexgnn_smoke_main1" --epochs 1 \
  --active 8 --games-per-epoch 8 --visits 32 --max-actions 40 --vbatch 8 \
  --train-steps-per-epoch 5 --batch 16 --eval-games 0 --soft-z-lambda 0 \
  --temperature-halflife 33 --temperature-floor 0.3 --eps 0.30 --device cuda 2>&1 | tail -22
