#!/usr/bin/env bash
# One-shot: repackage main_4 epoch_000025.pt into the WEIGHTS-ONLY soak-init
# {"meta","model"} shape expected by configs/hexfield_eq_main_10.toml
# (initialize_from). Mirrors _main4_repackage.sh with the SAME arch env
# (a5_lut2) so build_soak_init's meta-vs-env cross-check passes and the ep25
# weights load shape-for-shape.
#
# This drops the optimizer state / train_state: main_10 starts with a fresh
# optimizer, fresh replay window, and the epoch counter at 0 (initialize_from,
# NOT resume_from). Run this ONCE from WSL before the first main_10 launch.
set -euo pipefail
ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main}"
SRC_CKPT="${SRC_CKPT:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4/checkpoints/epoch_000025.pt}"
OUT_DIR="${OUT_DIR:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main10_prefit/main4ep25}"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "ABORT: source checkpoint not found: $SRC_CKPT" >&2
  exit 1
fi

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_raytap_a5_lut2.env"
EQ_LADDER_REPO="$ROOT"
set +a

mkdir -p "$OUT_DIR"
/root/.venvs/hexgt-build/bin/python -u "$ROOT/scripts/eq_ladder_runner.py" \
  --repackage a5 \
  --arm-dir "$OUT_DIR" \
  --ckpt "$SRC_CKPT" \
  --out "$OUT_DIR/soak_init.pt" \
  --weights raw
echo "--- repackage exit=$? ---"
ls -la "$OUT_DIR/"
