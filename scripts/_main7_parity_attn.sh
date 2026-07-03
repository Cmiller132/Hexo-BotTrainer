#!/usr/bin/env bash
# Parity battery for the bespoke Triton attention kernel (HEXFIELD_TRITON_ATTN)
# and the layout-driven trunk rewrite, against a FROZEN state snapshot (the
# live run keeps writing .hxr, so the mtime-ordered state battery must come
# from a static dir).
#
#  A  worktree pkg (old unrolled trunk), triton_conv+flex_pair  -> save base
#  B  dev pkg      (loop trunk),        same flags              -> tol 1e-6
#  C  dev pkg + TRITON_ATTN                                      -> tol 3e-3
#  D  dev pkg + SERVE_HALF save; + SERVE_HALF + TRITON_ATTN      -> tol 3e-3
set -eu
cd /mnt/e/Hexo-BotTrainer-hexgt
PY=/root/.venvs/hexgt-build/bin/python

FROZEN=/tmp/main7_ref_run
if [ ! -d "$FROZEN/selfplay" ]; then
  mkdir -p "$FROZEN/selfplay"
  ls -t /mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/selfplay/*.hxr | head -200 \
    | xargs -I{} cp -p {} "$FROZEN/selfplay/"
  echo "frozen $(ls "$FROZEN/selfplay" | wc -l) hxr files"
fi

export HEXFIELD_CHANNELS=128 HEXFIELD_SUPPORT_RADIUS=4 HEXFIELD_SERVE_FLEX=1
export HEXFIELD_REF_RUN=$FROZEN
export HEXFIELD_REF_CKPT=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python
WT=/mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python

echo '--- A: baseline save (WORKTREE pkg, triton_conv+flex_pair) ---'
HEXFIELD_REF_PKGROOT=$WT HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 \
  "$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_attn_base.npz
echo '--- B: dev pkg, same flags (trunk-loop identity, tol 1e-6) ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_attn_base.npz 1e-6
echo '--- C: dev pkg + TRITON_ATTN (kernel vs flex, tol 3e-3) ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_TRITON_ATTN=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_attn_base.npz 3e-3
echo '--- D1: dev pkg + SERVE_HALF baseline save ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 \
  "$PY" scripts/_hexfield_serve_ref.py save /tmp/m7_attn_half.npz
echo '--- D2: + TRITON_ATTN under SERVE_HALF (tol 3e-3) ---'
HEXFIELD_TRITON_CONV=1 HEXFIELD_FLEX_PAIR=1 HEXFIELD_SERVE_HALF=1 HEXFIELD_TRITON_ATTN=1 \
  "$PY" scripts/_hexfield_serve_ref.py check /tmp/m7_attn_half.npz 3e-3
echo ALL DONE
