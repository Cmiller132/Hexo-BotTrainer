#!/usr/bin/env bash
# Wave-1 serve pos/s per arm (throwaway wrapper): sources the arm env + the
# production fast serve profile (mirroring _hexfield_eq_supervise_main1.sh
# defaults), then runs raytap_serve_throughput.py on the arm's repackaged
# soak_init-style checkpoint (falls back to the raw prefit checkpoint).
# Usage: _raytap_wave1_poss.sh <arm_name> [visits]
set -uo pipefail
ARM="${1:?arm name}"
VISITS="${2:-512}"
WT=/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/raytap-phase-r
ROOT=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_raytap_wave1
cd "$WT"
source /root/.venvs/hexgt-build/bin/activate
set -a
source "scripts/prefit_env/hexfield_eq_${ARM}.env"
# Production fast serve profile (fast-serve fix bundle, 2026-07-09).
HEXFIELD_SERVE_FLEX=1
HEXFIELD_FLEX_PAIR=1
HEXFIELD_TRITON_CONV=1
HEXFIELD_TRITON_ATTN=1
HEXFIELD_TRITON_CONV_LN=1
HEXFIELD_SERVE_HALF=1
HEXFIELD_RUST_PACK=1
HEXFIELD_COPY_STREAM=1
HEXFIELD_CUDA_GRAPHS=1
HEXFIELD_EQ_TRITON_RAY=1
set +a
export PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python
CKPT="$ROOT/$ARM/soak_init.pt"
[[ -f "$CKPT" ]] || CKPT="$ROOT/$ARM/checkpoint_epoch0.pt"
python scripts/raytap_serve_throughput.py "$CKPT" "$VISITS" 16 20
