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

# Dashboard bridge watchdog: keep the read-only :8080 mirror bridge alive for the
# LIFE OF THE RUN. The bridge is a CHILD of this persistent supervisor (not a
# transient wsl.exe relay), so it survives any dispatch session ending — fixing
# the recurring "bridge keeps dying" problem; this loop also restarts it within
# ~30s if it exits. Single-instance: it only launches when no _dashboard_bridge.py
# is already running, so it never double-writes the mirror. Inherits the
# supervisor's PYTHONPATH/venv so the bridge's imports resolve.
BRIDGE_SCRIPT="$ROOT/_dashboard_bridge.py"; BRIDGE_WATCH_PID=""
if [[ -f "$BRIDGE_SCRIPT" ]]; then
  ( while :; do
      if ! pgrep -f "_dashboard_bridge.py" >/dev/null 2>&1; then
        echo "[$(date -u +%FT%TZ)] (re)starting dashboard bridge" >> "$RUNDIR/bridge.log"
        "$PY" -u "$BRIDGE_SCRIPT" >> "$RUNDIR/bridge.log" 2>&1 &
      fi
      sleep 30
    done ) & BRIDGE_WATCH_PID=$!
  log "bridge watchdog started (pid=$BRIDGE_WATCH_PID -> $BRIDGE_SCRIPT)"
fi

# GPU telemetry sampler (1 Hz) for crash forensics: temp/power/clocks/util/mem +
# clocks_throttle_reasons.active (directly flags sw/hw thermal slowdown, power cap,
# or reliability throttle). Appends to a CSV in the run dir; a CHILD of this
# persistent supervisor, restarted within ~30s if it dies. So the seconds before a
# crash are on record -> thermal/power vs a clean transient is distinguishable.
TELEM_CSV="$RUNDIR/gpu_telemetry.csv"; TELEM_WATCH_PID=""
( while :; do
    if ! pgrep -f "query-gpu=timestamp,temperature" >/dev/null 2>&1; then
      echo "[$(date -u +%FT%TZ)] (re)starting gpu telemetry sampler" >> "$RUNDIR/gpu_telemetry.err"
      nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used,clocks_throttle_reasons.active \
        --format=csv -l 1 >> "$TELEM_CSV" 2>>"$RUNDIR/gpu_telemetry.err" &
    fi
    sleep 30
  done ) & TELEM_WATCH_PID=$!
log "gpu telemetry sampler watchdog started (pid=$TELEM_WATCH_PID -> $TELEM_CSV)"

trap 'rm -f "$LOCK"; kill $WATCH_PID $BRIDGE_WATCH_PID $TELEM_WATCH_PID 2>/dev/null; pkill -f "_dashboard_bridge.py" 2>/dev/null; pkill -f "query-gpu=timestamp,temperature" 2>/dev/null' EXIT

rl_epoch(){ # highest completed rl_epoch from epoch checkpoints, else -1
  local f; f=$(ls -1 "$CKPTS"/hexgt_rl_epoch*.pt 2>/dev/null | sort -V | tail -1)
  [[ -z "$f" ]] && { echo -1; return; }
  basename "$f" | sed -E 's/hexgt_rl_epoch0*([0-9]+)\.pt/\1/'
}

capture_crash_diag(){ # $1=stamp $2=err_log $3=code $4=uptime — full forensic bundle
  # Make the next crash ATTRIBUTABLE without guessing: bundle the run-log error,
  # the 1Hz GPU telemetry window just before the crash (temp/power/throttle), the
  # NVIDIA error/Xid snapshot, and the Windows Event Log (TDR 4101 / nvlddmkm /
  # WHEA) around the crash time. -> TDR-only (4101, flat telemetry) vs thermal
  # (temp/throttle spike) vs power (power cap) vs a specific Xid hardware fault.
  local stamp="$1" err="$2" code="$3" up="$4"
  local dir="$CRASH/crash_diag_$stamp"; mkdir -p "$dir"
  { echo "stamp=$stamp exit=$code uptime=${up}s host_utc=$(date -u +%FT%TZ)";
    echo "--- driver err tail ---"; tail -n 60 "$err" 2>/dev/null;
    echo "--- rl_train.log tail ---"; tail -n 40 "$RUNDIR/rl_train.log" 2>/dev/null;
  } > "$dir/error.txt" 2>&1
  tail -n 150 "$TELEM_CSV" > "$dir/gpu_telemetry_window.csv" 2>/dev/null  # ~last 150s
  nvidia-smi -q -d ERRORS > "$dir/nvidia_smi_errors.txt" 2>&1             # Xid/ECC snapshot
  # Windows Event Log: TDR (4101) + NVIDIA/Display/WHEA in the last 20 min (single-
  # quoted so $ goes to PowerShell, not bash). Harmless if interop is unavailable.
  powershell.exe -NoProfile -Command '$s=(Get-Date).AddMinutes(-20); Get-WinEvent -FilterHashtable @{LogName="System"; StartTime=$s} -ErrorAction SilentlyContinue | Where-Object { $_.Id -eq 4101 -or $_.ProviderName -match "nvlddmkm|Display|WHEA" } | Select-Object TimeCreated,ProviderName,Id,LevelDisplayName | Format-Table -AutoSize | Out-String -Width 200' > "$dir/win_eventlog.txt" 2>&1
  log "CRASH DIAG captured -> $dir (telemetry window + nvidia errors + win eventlog)"
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
  # NON-ZERO EXIT = crash: capture the full forensic bundle (telemetry window, Xid,
  # Windows TDR/Display events) before relaunching, so it is fully attributable.
  capture_crash_diag "$stamp" "$err" "$code" "$up"
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
