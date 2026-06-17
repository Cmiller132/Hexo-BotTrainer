#!/usr/bin/env bash
# Drive the eval-v2 advanced ladder (multistage_eval) over a set of candidate
# epochs, SEQUENTIALLY, in the non-live hexo-bottrainer-wsl venv. Resumable:
# re-running skips per-(epoch,opponent) edges already in the pool. One process =
# no eval_pool.json write race, no GPU thrash. PURE EVAL (writes only the pool +
# its own diagnostics; never touches the training run).
#
# Usage: bash _hexfield_learning_ladder.sh "<epochs csv>" <full_visits> <budget>
#   e.g. bash _hexfield_learning_ladder.sh "10,20,25" 256 64
set -u
REPO=/mnt/e/Hexo-BotTrainer-hexgt
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1
CFG=$REPO/configs/hexfield_main_1.toml
VENV=/root/.venvs/${4:-hexo-bottrainer-wsl}
CK=$RUN/checkpoints
EPOCHS="${1:-10,20,25}"
FULL_VISITS="${2:-256}"
BUDGET="${3:-64}"
export PYTHONPATH="$REPO/packages/hexfield/python:$REPO/packages/hexo_engine/python:$REPO/packages/hexo_runner/python:$REPO/packages/hexo_models/python:$REPO/packages/hexo_utils/python:$REPO/packages/dense_cnn_restnet/python"
PY="$VENV/bin/python"
RUNNER="$REPO/scripts/_hexfield_run_multistage_eval.py"
echo "[ladder] epochs=$EPOCHS full_visits=$FULL_VISITS budget=$BUDGET  $(date -u +%FT%TZ)"
IFS=',' read -ra EPS <<< "$EPOCHS"
for ep in "${EPS[@]}"; do
  ckpt="$CK/epoch_$(printf '%06d' "$ep").pt"
  if [ ! -f "$ckpt" ]; then echo "[ladder] SKIP ep$ep (no checkpoint $ckpt)"; continue; fi
  echo "[ladder] ===== candidate ep$ep ($ckpt) ====="
  "$PY" "$RUNNER" "$RUN" "$ckpt" --epoch "$ep" --config "$CFG" \
    --full-visits "$FULL_VISITS" --budget "$BUDGET" --no-sealbot --parts --resume \
    || echo "[ladder] ep$ep runner exit=$?"
  echo "[ladder] ===== done ep$ep  $(date -u +%FT%TZ) ====="
done
echo "[ladder] ALL DONE $(date -u +%FT%TZ)"
