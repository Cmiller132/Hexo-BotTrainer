#!/usr/bin/env bash
# WSL-side supervisor for the dense_cnn target_96x8 run. Identical guardrails to
# supervise_target_64x4_wsl.sh (resume-from-latest, circuit breaker, completion
# guard, crash artifacts, RAM watchdog, single-instance lock); only the run name,
# config, and run dir differ. Launches under the WSL venv so TensorRT FP16 engages.
set -uo pipefail

ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer}"
VENV="${VENV:-/root/.venvs/hexo-bottrainer-wsl}"
CONFIG="${CONFIG:-$ROOT/configs/dense_cnn_model1_target_96x8.toml}"
RUNDIR="${RUNDIR:-$ROOT/runs/dense_cnn_model1_target_96x8}"
SEALBOT="${SEALBOT_PATH:-/mnt/e/SealBot}"
DIAG="$RUNDIR/diagnostics"; CKPTS="$RUNDIR/checkpoints"; CRASH="$DIAG/crash_artifacts"
HALT="$DIAG/supervisor_halted.flag"; DONE="$DIAG/supervisor_completed.flag"
SUPLOG="$DIAG/supervisor_wsl.log"; LOCK="$DIAG/supervisor_wsl.lock"
PY="$VENV/bin/python"

FAST_CRASH_SECONDS=180; MAX_CONSEC_FAST=3; MAX_PER_HOUR=6; MAX_NO_PROGRESS=5; MIN_FREE_RAM_GB=4

export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
export SEALBOT_PATH="$SEALBOT"
export PYTHONPATH="$PYTHONPATH:$SEALBOT:$SEALBOT/best"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$DIAG" "$CKPTS" "$CRASH"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another WSL supervisor running (pid $(cat "$LOCK")))"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; exit 1; }
rm -f "$DONE"

loop_epochs(){ "$PY" - "$CONFIG" <<'PY'
import sys,tomllib; print(int(tomllib.load(open(sys.argv[1],"rb")).get("loop",{}).get("epochs",1)))
PY
}
latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }
epoch_of(){ local f="$1"; [[ -z "$f" ]] && { echo 0; return; }; basename "$f" | sed -E 's/epoch_0*([0-9]+)\.pt/\1/'; }
set_resume(){
  "$PY" - "$CONFIG" "$1" <<'PY'
import sys,re,pathlib
cfg=pathlib.Path(sys.argv[1]); rf=sys.argv[2].replace("\\","/"); t=cfg.read_text()
t, n = re.subn(r'(?m)^[ \t]*resume_from[ \t]*=.*$', f'resume_from = "{rf}"', t)
if n == 0:
    t = re.sub(r'(?m)^(\[checkpoint\][ \t]*)$', r'\1\nresume_from = "'+rf+'"', t, count=1)
cfg.write_text(t)
PY
}

LOOP_EPOCHS="$(loop_epochs)"
log "WSL SUPERVISOR start (pid=$$) run=dense_cnn_model1_target_96x8 config=$CONFIG epochs=$LOOP_EPOCHS"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr OR no-progress x${MAX_NO_PROGRESS} -> halt; venv=$VENV; TRT engages via config inference_use_tensorrt=true"

( while :; do
    fg=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    echo "{\"t\":\"$(date -u +%FT%TZ)\",\"free_ram_gb\":$fg}" >> "$DIAG/watch_wsl.jsonl"
    awk -v f="$fg" -v m="$MIN_FREE_RAM_GB" 'BEGIN{exit !(f<m)}' && echo "[$(date -u +%FT%TZ)] WARN free RAM ${fg}GB < ${MIN_FREE_RAM_GB}GB" >> "$SUPLOG"
    sleep 15
  done ) & WATCH_PID=$!
trap 'rm -f "$LOCK"; kill $WATCH_PID 2>/dev/null' EXIT

declare -a crash_times=(); consec_fast=0; last_prog_epoch="$(epoch_of "$(latest_ckpt)")"; no_progress=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then set_resume "$lc"; log "resume_from -> $lc (start epoch $(( $(epoch_of "$lc")+1 )))"
  else log "no epoch checkpoint; launch uses initialize_from (SealBot prefit) -> bootstrapped"; fi

  stamp="$(date -u +%Y%m%d_%H%M%S)"; out="$DIAG/trainer.$stamp.out.log"; err="$DIAG/trainer.$stamp.err.log"
  t0=$(date +%s)
  "$PY" -u -m hexo_train.cli.train_model "$CONFIG" >"$out" 2>"$err" &
  cpid=$!; echo "$cpid" > "$DIAG/trainer_wsl.pid"
  log "LAUNCH pid=$cpid out=$out err=$err"
  wait "$cpid"; code=$?; t1=$(date +%s); up=$((t1-t0))
  log "EXIT pid=$cpid code=$code uptime=${up}s"
  { echo "exit=$code uptime=${up}s stamp=$stamp"; echo "--- err tail ---"; tail -40 "$err"; } > "$CRASH/crash.$stamp.txt" 2>/dev/null

  lc="$(latest_ckpt)"; le="$(epoch_of "$lc")"
  if (( le+1 > LOOP_EPOCHS )); then echo "completed through epoch $le" > "$DONE"; log "COMPLETED through epoch $le (loop.epochs=$LOOP_EPOCHS)"; break; fi
  if (( le > last_prog_epoch )); then last_prog_epoch=$le; no_progress=0; else no_progress=$((no_progress+1)); fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker state: consecFast=$consec_fast crashesLastHour=${#crash_times[@]} noProgress=$no_progress/$MAX_NO_PROGRESS (latest epoch $le)"
  if (( no_progress >= MAX_NO_PROGRESS || consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: noProgress=$no_progress consecFast=$consec_fast crashesLastHour=${#crash_times[@]} (latest epoch $le)" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest)"; sleep 2
done
log "WSL SUPERVISOR exit."
