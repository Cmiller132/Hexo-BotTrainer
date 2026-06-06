#!/usr/bin/env bash
# Lean unattended supervisor for the FROM-SCRATCH dense_cnn_rl_main1 run (Model 1
# 96x8 @ 512 sims), the head-to-head baseline vs hexgnn. Mirrors the hexgnn
# supervisor's shape (auto-relaunch + circuit breaker + single-instance lock +
# halt flag) but drives the dense_cnn config-driven CLI (hexo_train.cli.train_model)
# instead of the hexgnn driver. NO resource watchdog (owner: no watchers).
#
# Run dir lives under /mnt/e/Hexo-BotTrainer/runs so the :8080 dashboard (cwd there)
# renders it directly — no bridge needed. First launch is from-scratch (the config
# has no resume_from/initialize_from); a crash-relaunch injects resume_from=<latest
# epoch_*.pt> into a throwaway config copy so the pipeline resumes.
set -uo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt}"
VENV="${VENV:-/root/.venvs/hexgt-build}"
CONFIG="${CONFIG:-$ROOT/configs/dense_cnn_rl_main1.toml}"
RUNDIR="${RUNDIR:-/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1}"
SEALBOT="${SEALBOT_PATH:-/mnt/e/SealBot}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
PY="$VENV/bin/python"
FAST_CRASH_SECONDS=300; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export SEALBOT_PATH="$SEALBOT"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python:$SEALBOT:$SEALBOT/best"

mkdir -p "$RUNDIR" "$CKPTS"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another dense_cnn supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"
trap 'rm -f "$LOCK"' EXIT

latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }

log "dense_cnn RL SUPERVISOR start (pid=$$) run=$RUNDIR config=$CONFIG"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then
    USE="$RUNDIR/_resume_config.toml"
    # inject resume_from right under [checkpoint]; pipeline resumes from this ckpt
    awk -v c="$lc" '/^\[checkpoint\]/{print; print "resume_from = \"" c "\""; next} {print}' "$CONFIG" > "$USE"
    log "RESUME from $(basename "$lc")"
  else
    USE="$CONFIG"; log "FIRST LAUNCH (from scratch, random init)"
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
log "dense_cnn RL SUPERVISOR exit."
