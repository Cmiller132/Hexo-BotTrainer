#!/usr/bin/env bash
# Unattended supervisor for the hexgt BC-seeded RL self-play run. Mirrors the
# dense_cnn supervisor's guardrails (auto-relaunch, circuit breaker, RAM
# watchdog, completion guard, crash artifacts, single-instance lock). The RL
# driver (_rl_train.py) is itself resumable (hexgt_rl_latest.pt carries model +
# optimizer + train-state + rl_epoch), so a relaunch just continues; the
# supervisor does not edit any config.
set -uo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt}"
VENV="${VENV:-/root/.venvs/hexgt-build}"
RUNDIR="${RUNDIR:-$ROOT/runs/hexgt_rl}"
SEALBOT="${SEALBOT_PATH:-/mnt/e/SealBot}"
EPOCHS="${EPOCHS:-40}"
GAMES_PER_EPOCH="${GAMES_PER_EPOCH:-64}"
EVAL_EVERY="${EVAL_EVERY:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
CRASH="$RUNDIR/crash_artifacts"
PY="$VENV/bin/python"

FAST_CRASH_SECONDS=180; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8; MAX_NO_PROGRESS=4; MIN_FREE_RAM_GB=4

export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export SEALBOT_PATH="$SEALBOT"
export PYTHONPATH="$PYTHONPATH:$SEALBOT:$SEALBOT/best"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

mkdir -p "$RUNDIR" "$CKPTS" "$CRASH"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another RL supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"

# RAM watchdog (records a trend + warns; the driver checkpoints every epoch so a
# kill loses at most one epoch).
( while :; do
    fg=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    echo "{\"t\":\"$(date -u +%FT%TZ)\",\"free_ram_gb\":$fg}" >> "$RUNDIR/watch_ram.jsonl"
    awk -v f="$fg" -v m="$MIN_FREE_RAM_GB" 'BEGIN{exit !(f<m)}' && \
      echo "[$(date -u +%FT%TZ)] WARN free RAM ${fg}GB < ${MIN_FREE_RAM_GB}GB" >> "$SUPLOG"
    sleep 20
  done ) & WATCH_PID=$!
trap 'rm -f "$LOCK"; kill $WATCH_PID 2>/dev/null' EXIT

rl_epoch(){ # highest completed rl_epoch from epoch checkpoints, else -1
  local f; f=$(ls -1 "$CKPTS"/hexgt_rl_epoch*.pt 2>/dev/null | sort -V | tail -1)
  [[ -z "$f" ]] && { echo -1; return; }
  basename "$f" | sed -E 's/hexgt_rl_epoch0*([0-9]+)\.pt/\1/'
}

log "RL SUPERVISOR start (pid=$$) run=$RUNDIR epochs=$EPOCHS games/epoch=$GAMES_PER_EPOCH eval_every=$EVAL_EVERY"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr OR no-progress x${MAX_NO_PROGRESS} -> halt"

declare -a crash_times=(); consec_fast=0; last_prog="$(rl_epoch)"; no_progress=0
while :; do
  cur="$(rl_epoch)"
  if (( cur + 1 >= EPOCHS )); then echo "completed through rl_epoch $cur" > "$DONE"; log "COMPLETED through rl_epoch $cur (epochs=$EPOCHS)"; break; fi

  stamp="$(date -u +%Y%m%d_%H%M%S)"; out="$RUNDIR/driver.$stamp.out.log"; err="$RUNDIR/driver.$stamp.err.log"
  t0=$(date +%s)
  log "LAUNCH (resume auto from latest; cur rl_epoch=$cur) out=$out"
  "$PY" -u "$ROOT/scripts/_rl_train.py" \
      --out-dir "$RUNDIR" --epochs "$EPOCHS" --games-per-epoch "$GAMES_PER_EPOCH" \
      --eval-every "$EVAL_EVERY" --sealbot $EXTRA_ARGS >"$out" 2>"$err" &
  cpid=$!; echo "$cpid" > "$RUNDIR/driver.pid"
  wait "$cpid"; code=$?; t1=$(date +%s); up=$((t1-t0))
  log "EXIT pid=$cpid code=$code uptime=${up}s"
  { echo "exit=$code uptime=${up}s stamp=$stamp"; echo "--- err tail ---"; tail -50 "$err"; } > "$CRASH/crash.$stamp.txt" 2>/dev/null

  cur="$(rl_epoch)"
  if (( code == 0 )); then echo "driver exited 0 at rl_epoch $cur" > "$DONE"; log "DONE: driver exited 0 (rl_epoch $cur)"; break; fi
  if (( cur + 1 >= EPOCHS )); then echo "completed through rl_epoch $cur" > "$DONE"; log "COMPLETED through rl_epoch $cur"; break; fi

  if (( cur > last_prog )); then last_prog=$cur; no_progress=0; else no_progress=$((no_progress+1)); fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker state: consecFast=$consec_fast crashesLastHour=${#crash_times[@]} noProgress=$no_progress/$MAX_NO_PROGRESS (rl_epoch $cur)"
  if (( no_progress >= MAX_NO_PROGRESS || consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: noProgress=$no_progress consecFast=$consec_fast crashesLastHour=${#crash_times[@]} (rl_epoch $cur)" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest)"; sleep 3
done
log "RL SUPERVISOR exit."
