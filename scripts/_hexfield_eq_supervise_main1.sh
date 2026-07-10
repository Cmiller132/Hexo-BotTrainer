#!/usr/bin/env bash
# Unattended supervisor for the hexfield_eq lineage (D6-equivariant rewrite) —
# a dedicated copy of scripts/_hexfield_supervise_main1.sh with the
# hexfield_eq PYTHONPATH and eq-run defaults. The live hexfield supervisor
# script is deliberately NOT shared: its PYTHONPATH hardcodes
# packages/hexfield/python, and the live main-lineage services exec it.
#
# Same auto-relaunch + circuit breaker + single-instance lock + halt flag;
# drives hexo_train.cli.train_model and RESUMES from the latest epoch
# checkpoint (resume_from injected into [checkpoint]).
#
# Referenced by scripts/systemd/hexfield-eq-supervisor-1.service, which owns
# the LOAD-BEARING HEXFIELD_EQ_* architecture env block.
set -uo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt}"
VENV="${VENV:-/root/.venvs/hexgt-build}"
CONFIG="${CONFIG:-$ROOT/configs/hexfield_eq_main_1.toml}"
RUNDIR="${RUNDIR:-/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_1}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
PY="$VENV/bin/python"
FAST_CRASH_SECONDS=300; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# SealBot zero-point reachability (multi_stage_eval sealbot_enabled=true):
# without SEALBOT_PATH the eval fail-opens and silently drops the zero-point.
export SEALBOT_PATH="${SEALBOT_PATH:-/mnt/e/SealBot}"
# hexfield_eq FIRST (this lineage's package); dense_cnn_restnet = legacy-shard
# oracle adapter; hexo_strix = the Strix eval anchor (not pip-installed — absent
# from PYTHONPATH the in-run Strix opponent import fails and the BT anchor
# silently falls back to SealBot).
# hexo_utils/hexo_train/hexo_engine/hexo_models/hexo_runner are NOT installed
# in the hexgt-build venv (unlike the live lineage's venv) — carry them on
# PYTHONPATH explicitly or the trainer entry dies at import (hexo_utils
# ModuleNotFoundError, 2026-07-09 launch crash).
export PYTHONPATH="$ROOT/packages/hexfield_eq/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/dense_cnn_restnet/python:$ROOT/packages/hexo_strix/python"
# GPU/host overlap in the self-play serve loop (bit-identical, parity-gated in
# the main lineage; mechanism-generic). Set to 0 to fall back to sync.
export HEXFIELD_ASYNC_EVAL="${HEXFIELD_ASYNC_EVAL:-1}"
# ---- FAST SERVE PROFILE (flipped on 2026-07-09) ----------------------------
# The conservative eager launch measured 4.2 pos/s vs main_11's ~50; the
# serve parity gates (3e-3) were re-run green under this exact flag set
# (tests/test_hexfield_eq_{triton_ray,ray_block,raylen_parity,serve}.py at the
# arm-4 arch) before the flip. Every knob stays env-overridable (=0 reverts
# to the eager reference path). Checklist §5.4; never HEXFIELD_CONV_FP8.
export HEXFIELD_SERVE_FLEX="${HEXFIELD_SERVE_FLEX:-1}"
export HEXFIELD_FLEX_PAIR="${HEXFIELD_FLEX_PAIR:-1}"
export HEXFIELD_TRITON_CONV="${HEXFIELD_TRITON_CONV:-1}"
export HEXFIELD_TRITON_ATTN="${HEXFIELD_TRITON_ATTN:-1}"
export HEXFIELD_TRITON_CONV_LN="${HEXFIELD_TRITON_CONV_LN:-1}"
# Split ray-tap serve path (2026-07-10): taps kernel + cuBLAS GEMM + LN
# kernel, replacing K1 for equipped convs. Deterministic micro-bench: faster
# than K1 at every Npad 179..1396 (1.4x small/mid; K1 regresses past
# reference at Npad~1396 while the split holds 2.7ms/conv); serve parity
# suites green under this exact flag set; =0 reverts to K1.
export HEXFIELD_TRITON_RAYTAP7="${HEXFIELD_TRITON_RAYTAP7:-1}"
# Gathered L-block ray attention (spec D-S36/D-S37) — the eq-specific kernel;
# A-block kernels never apply to L, so without this the L blocks fall back to
# the materialized (B, 6, N, N) bias path that capped serve at ~4 pos/s.
export HEXFIELD_EQ_TRITON_RAY="${HEXFIELD_EQ_TRITON_RAY:-1}"
# CUDA-graph capture/replay for the serve forward (inference._GraphCache).
# Legal alongside the ray kernel since 2026-07-09: the gather-index build is
# sync-free (sort/searchsorted join). Serve gate 7/7 with this exact stack;
# live: mid-epoch marginal ~15 -> ~21 pos/s (the serve loop was submit-bound).
export HEXFIELD_CUDA_GRAPHS="${HEXFIELD_CUDA_GRAPHS:-1}"
export HEXFIELD_SERVE_HALF="${HEXFIELD_SERVE_HALF:-1}"
export HEXFIELD_RUST_PACK="${HEXFIELD_RUST_PACK:-1}"
export HEXFIELD_COPY_STREAM="${HEXFIELD_COPY_STREAM:-1}"
# Train-side: flex attention (no S^2 bias transient; see trainer._pair_budget
# note) + compiled train forward. Both main-lineage proven.
export HEXFIELD_TRAIN_FLEX="${HEXFIELD_TRAIN_FLEX:-1}"
export HEXFIELD_TRAIN_COMPILE="${HEXFIELD_TRAIN_COMPILE:-1}"
# Deferred-decode (bit-identical decode scheduling). Set to 0 to keep the
# device syncs inside submit.
export HEXFIELD_DEFER_DECODE="${HEXFIELD_DEFER_DECODE:-1}"
# Model-side support radius: HEXFIELD_EQ_SUPPORT_RADIUS (EQ-namespaced), set by
# the systemd unit (=4 — must match the prefit corpus / winner checkpoint).

mkdir -p "$RUNDIR" "$CKPTS"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another hexfield_eq supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"
trap 'rm -f "$LOCK"' EXIT

latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }

log "hexfield_eq SUPERVISOR start (pid=$$) run=$RUNDIR config=$CONFIG"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then
    USE="$RUNDIR/_resume_config.toml"
    # Inject resume_from right after [checkpoint]; hexo_train prefers it over
    # initialize_from, and the loader then loads model+optimizer+epoch.
    awk -v c="$lc" '/^\[checkpoint\]/{print; print "resume_from = \"" c "\""; next} {print}' "$CONFIG" > "$USE"
    log "RESUME from $(basename "$lc")"
  else
    USE="$CONFIG"; log "FIRST LAUNCH (init per config: initialize_from BC prefit)"
  fi
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  t0=$(date +%s)
  log "LAUNCH out=$RUNDIR/train.$stamp.out.log"
  "$PY" -u -m hexo_train.cli.train_model "$USE" >"$RUNDIR/train.$stamp.out.log" 2>&1 &
  cpid=$!; echo "$cpid" > "$RUNDIR/driver.pid"
  wait "$cpid"; code=$?; t1=$(date +%s); up=$((t1-t0))
  log "EXIT pid=$cpid code=$code uptime=${up}s"
  if (( code != 0 )); then
    tail -n 40 "$RUNDIR/train.$stamp.out.log" 2>/dev/null \
      | grep -E 'Error|Traceback|raise |CUDA|assert' | tail -n 3 \
      | while IFS= read -r line; do log "CRASH| $line"; done
  fi
  if (( code == 0 )); then echo "exit 0 at $(date -u +%FT%TZ)" > "$DONE"; log "DONE (exit 0)"; break; fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}"
  if (( consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest) in 3s"; sleep 3
done
log "hexfield_eq SUPERVISOR exit."
