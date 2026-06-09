#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
PY=/root/.venvs/hexgt-build/bin/python
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
echo "=== hexgt tests (n=3 + pruning) ==="
"$PY" -m pytest tests/test_hexgt_expand_train.py tests/test_hexgt_candidates.py -q 2>&1 | tail -8
echo ""
echo "=== COVERAGE sweep on RECENT 96x8 (read-only, completed epochs) ==="
"$PY" scripts/_validate_hexgt_candidates.py --runs /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay --max-games 40 --default-n 3 --recent 2>&1 | head -40
echo ""
echo "=== EXPAND prune-rate on RECENT 96x8 (n=2/3/4, threshold 0.15) ==="
"$PY" scripts/_validate_hexgt_expand.py --runs /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/selfplay --max-games 16 --radii 2 3 4 --recent --prune-threshold 0.15
