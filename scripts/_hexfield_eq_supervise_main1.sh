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
# GPU-health gating (2026-07-15). The 07-14 night crash chain: a driver-level
# CUDA context death (segfault in libc10_cuda.so) killed the trainer mid-eval,
# and the blind 3s relaunch went straight into the still-wedged dxg context and
# died again 588s in — one near-miss short of the fast-crash breaker HALTING
# the run overnight over a transient driver flake. Before every (re)launch,
# probe the GPU with a real CUDA matmul and wait (backoff, capped) until it
# passes; after a CUDA-signature crash, settle first — the wedged WSL dxg
# layer heals off-load. Probe waits and CUDA-signature crashes do NOT advance
# the consecutive-fast-crash breaker (the probe gate replaces it); the
# crashes-per-hour backstop still counts everything.
GPU_PROBE_TIMEOUT=120; GPU_SETTLE_SECONDS=60; GPU_PROBE_RETRY_MAX=600

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
# Coords-direct attention (2026-07-10): the attn_pair kernel with the bias
# row computed IN-KERNEL from coords — the (B, S, S) uint8 pair tensor and
# its once-per-forward build (measured 13.2 ms at B=96/S=862, more device
# time than all three A-blocks' attention math combined; its int intermediates
# also pinned CUDA-graph pool VRAM per captured key) are gone. Ply-60 MCTS
# A/B at the unchanged 3.8e7 ceiling: 4.41 -> 7.12 pos/s (+61%). Kernel
# parity identical to attn_pair's class (masked max-abs 0.002-0.004 vs the
# materialized reference at every probed shape); =0 reverts to attn_pair +
# the pair build. Do NOT pair this with a raised HEXFIELD_PAIR_CEILING —
# measured regression (see inference.py PAIR_CEILING note).
export HEXFIELD_TRITON_ATTN2="${HEXFIELD_TRITON_ATTN2:-1}"
# Gathered L-block ray attention (spec D-S36/D-S37) — the eq-specific kernel;
# A-block kernels never apply to L, so without this the L blocks fall back to
# the materialized (B, 6, N, N) bias path that capped serve at ~4 pos/s.
export HEXFIELD_EQ_TRITON_RAY="${HEXFIELD_EQ_TRITON_RAY:-1}"
# CUDA-graph capture/replay for the serve forward (inference._GraphCache).
# Legal alongside the ray kernel since 2026-07-09: the gather-index build is
# sync-free (sort/searchsorted join). Serve gate 7/7 with this exact stack;
# live: mid-epoch marginal ~15 -> ~21 pos/s (the serve loop was submit-bound).
export HEXFIELD_CUDA_GRAPHS="${HEXFIELD_CUDA_GRAPHS:-1}"
# 2026-07-11 diagnostic: warm-boundary earlyoom kills persist at ~26.8GB even
# with the graph-key cap FIRING at its default 96 — either 96 keys x
# ~50-70MB host cudaGraph exec each IS the ~7GB warm accumulation (cap too
# high to bind), or graphs are exonerated. 24 discriminates: if the warm
# selfplay->train peak drops ~4-6GB and the kills stop, tune the cap and
# keep it; if unchanged, look at dynamo guards / rust session growth next.
# Throughput cost expected small (the 24 hottest keys carry most groups).
export HEXFIELD_GRAPH_MAX_KEYS="${HEXFIELD_GRAPH_MAX_KEYS:-24}"
export HEXFIELD_SERVE_HALF="${HEXFIELD_SERVE_HALF:-1}"
export HEXFIELD_RUST_PACK="${HEXFIELD_RUST_PACK:-1}"
export HEXFIELD_COPY_STREAM="${HEXFIELD_COPY_STREAM:-1}"
# Train-side: flex attention (no S^2 bias transient; see trainer._pair_budget
# note) + compiled train forward. Both main-lineage proven.
export HEXFIELD_TRAIN_FLEX="${HEXFIELD_TRAIN_FLEX:-1}"
export HEXFIELD_TRAIN_COMPILE="${HEXFIELD_TRAIN_COMPILE:-1}"
# Fused Triton train-stream ray-tap aggregation (2026-07-13): retires the
# eager K2 island (was ~85% of train-step device time at main_3 shapes) for a
# fwd+bwd custom op that stays in-graph under the compiled trainer. Gated
# inside model.py (grad-enabled CUDA only; per-shape eager fallback). =0
# reverts to the eager K2 island with zero code change.
export HEXFIELD_TRITON_RAYTAP_TRAIN="${HEXFIELD_TRITON_RAYTAP_TRAIN:-1}"
# Deferred-decode (bit-identical decode scheduling). Set to 0 to keep the
# device syncs inside submit.
export HEXFIELD_DEFER_DECODE="${HEXFIELD_DEFER_DECODE:-1}"
# Model-side support radius: HEXFIELD_EQ_SUPPORT_RADIUS (EQ-namespaced), set by
# the systemd unit (=4 — must match the prefit corpus / winner checkpoint).
# 2026-07-11: lazy chunked train expansion at radius<8. The radius-4 run was
# silently routed to the legacy upfront branch (all 56k dense rows at once,
# ~21GB transient — THE warm-boundary earlyoom killer, py-spy-confirmed; the
# tolerate_off_legal radius gate had conflated legality semantics with the
# expansion strategy). 0 off-legal rows across the entire run, so the lazy
# path's strict contract holds; an off-legal row would raise -> supervised
# restart (never observed).
export HEXFIELD_EXPAND_LAZY="${HEXFIELD_EXPAND_LAZY:-1}"

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

# A real CUDA round-trip (context init + matmul + sync), not nvidia-smi: after
# the 07-14 wedge, light queries still succeeded while context work died.
gpu_probe(){
  timeout "$GPU_PROBE_TIMEOUT" "$PY" -c '
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(1)
a = torch.rand(2048, 2048, device="cuda")
s = float((a @ a).abs().sum().item())
torch.cuda.synchronize()
sys.exit(0 if s > 0 else 1)
' >/dev/null 2>&1
}

wait_for_gpu(){
  local delay=60 n=0
  until gpu_probe; do
    n=$((n+1))
    log "GPU PROBE failed (attempt $n) — GPU/driver not ready; retrying in ${delay}s (probe waits do not count as crashes)"
    sleep "$delay"
    delay=$(( delay * 2 )); (( delay > GPU_PROBE_RETRY_MAX )) && delay=$GPU_PROBE_RETRY_MAX
  done
  if (( n > 0 )); then log "GPU PROBE ok after $n failed attempt(s)"; else log "GPU PROBE ok"; fi
}

log "hexfield_eq SUPERVISOR start (pid=$$) run=$RUNDIR config=$CONFIG"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  wait_for_gpu
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
  gpu_sig=0
  if (( code != 0 )); then
    tail -n 40 "$RUNDIR/train.$stamp.out.log" 2>/dev/null \
      | grep -E 'Error|Traceback|raise |CUDA|assert' | tail -n 3 \
      | while IFS= read -r line; do log "CRASH| $line"; done
    if tail -n 80 "$RUNDIR/train.$stamp.out.log" 2>/dev/null \
        | grep -qE 'CUDA error|AcceleratorError|cudaError|libc10_cuda|CUDA driver error'; then
      gpu_sig=1
    fi
  fi
  if (( code == 0 )); then echo "exit 0 at $(date -u +%FT%TZ)" > "$DONE"; log "DONE (exit 0)"; break; fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( gpu_sig )); then
    log "CUDA-signature crash — settling ${GPU_SETTLE_SECONDS}s (wedged dxg heals off-load); consecFast unchanged, GPU probe gates the relaunch"
    sleep "$GPU_SETTLE_SECONDS"
  elif (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}"
  if (( consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest) in 3s"; sleep 3
done
log "hexfield_eq SUPERVISOR exit."
