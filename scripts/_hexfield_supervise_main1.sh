#!/usr/bin/env bash
# Unattended supervisor for hexfield_main_1 — mirrors _dc_restnet_supervise_main1.sh
# (auto-relaunch + circuit breaker + single-instance lock + halt flag), drives the
# config-driven CLI (hexo_train.cli.train_model), and RESUMES from the latest
# epoch checkpoint (resume_from injected into [checkpoint]; hexo_train prefers
# resume_from over initialize_from, and HexfieldCheckpointLoader loads
# model+optimizer+epoch when resume_from is set). Run dir is under
# /mnt/e/Hexo-BotTrainer/runs so the :8080 dashboard renders it directly.
# PYTHONPATH adds the untracked hexfield package (never installed in the venv,
# spec §5.1) + dense_cnn_restnet (the legacy-shard oracle adapter).
set -uo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt}"
VENV="${VENV:-/root/.venvs/hexgt-build}"
CONFIG="${CONFIG:-$ROOT/configs/hexfield_main_1.toml}"
RUNDIR="${RUNDIR:-/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
PY="$VENV/bin/python"
FAST_CRASH_SECONDS=300; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# expandable_segments eases caching-allocator fragmentation under the high
# GPU-memory pressure this run shows (~11.9/12.3 GiB).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# SealBot zero-point reachability: the in-run eval (multi_stage_eval, sealbot_enabled
# =true) resolves the SealBot binary via $SEALBOT_PATH (hexo_runner SealBotConfig).
# The Windows SEALBOT_PATH is invisible to the WSL trainer, so without this the eval
# FAIL-OPENS and silently drops the down-weighted zero-point edge. Binary lives at
# /mnt/e/SealBot/{current,best} (variant 'current').
export SEALBOT_PATH="${SEALBOT_PATH:-/mnt/e/SealBot}"
# Minimal PYTHONPATH proven by the original launch (hexo_* resolve from the venv).
export PYTHONPATH="$ROOT/packages/hexfield/python:$ROOT/packages/dense_cnn_restnet/python"
# GPU/host overlap in the self-play serve loop (2026-06-15): the forward is
# submitted (no device sync), the pre-backup prefetch select runs GIL-released
# while those kernels execute, then the forward is drained. Bit-identical to the
# sync path on identical inputs (parity-gated). Set to 0 to fall back to sync.
export HEXFIELD_ASYNC_EVAL="${HEXFIELD_ASYNC_EVAL:-1}"
# Serve-only FlexAttention (2026-06-16): the no-grad serve forward computes the
# rel-pos bias INSIDE the attention kernel (score_mod) instead of materializing the
# (B,heads,S,S) bias — the profiled ~68% of the serve forward. Parity-gated
# (flex-OFF bit-exact; flex-ON max|dvalue| 1.24e-3 within the shipped 3e-3 fp16
# tolerance), dynamic-shape clean (1 compile, 0 CantSplit/recompiles). Halves the
# forward + ~20x less serve VRAM; ~1.2-1.26x end-to-end pos/s (host-bound after).
# One-time ~10s flex compile on the first serve forward. Training/grad path
# untouched (fires only under no_grad). Set to 0 to revert to the materialized bias.
export HEXFIELD_SERVE_FLEX="${HEXFIELD_SERVE_FLEX:-1}"

mkdir -p "$RUNDIR" "$CKPTS"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another hexfield supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"
trap 'rm -f "$LOCK"' EXIT

latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }

log "hexfield SUPERVISOR start (pid=$$) run=$RUNDIR config=$CONFIG"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then
    USE="$RUNDIR/_resume_config.toml"
    # Inject resume_from right after [checkpoint]; hexo_train prefers it over
    # initialize_from, and the hexfield loader then loads model+optimizer+epoch.
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
log "hexfield SUPERVISOR exit."
