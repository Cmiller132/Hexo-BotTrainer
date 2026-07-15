#!/usr/bin/env bash
# One-shot: repackage the main_2 A5 prefit checkpoint into the soak-init
# {"meta","model"} form expected by configs/hexfield_eq_main_2.toml
# (initialize_from). Mirrors eq_ladder_runner.repackage_cmd with the A5 arch
# env sourced and the (now raytap-capable) main tree as EQ_LADDER_REPO.
set -euo pipefail
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
ARM_DIR=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main2_prefit/a5

set -a
source "$ROOT/scripts/prefit_env/hexfield_eq_raytap_a5.env"
EQ_LADDER_REPO="$ROOT"
set +a

/root/.venvs/hexgt-build/bin/python -u "$ROOT/scripts/eq_ladder_runner.py" \
  --repackage a5 \
  --arm-dir "$ARM_DIR" \
  --ckpt "$ARM_DIR/checkpoint_epoch0.pt" \
  --out "$ARM_DIR/soak_init.pt" \
  --weights raw
echo "--- repackage exit=$? ---"
ls -la "$ARM_DIR/"
