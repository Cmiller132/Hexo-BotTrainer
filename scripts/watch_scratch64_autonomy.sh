#!/usr/bin/env bash
# Autonomy-aware watcher for the supervised dense_cnn scratch_64 run.
# The host-side supervisor (supervise_scratch64.ps1) owns relaunch + capture, so
# a brief gap during relaunch is NORMAL and is NOT reported as a crash. Emits one
# stdout line per notable event (Monitor -> chat notification):
#   - new minidump in crashdumps/
#   - new fault signature in the (rolling) trainer.*.err.log
#   - supervisor lifecycle: EXIT / RELAUNCH / CAPTURE / COMPLETED / HALT
#   - circuit-breaker halt flag appears
#   - SealBot eval results per epoch (win rate) and new epoch checkpoints
#   - long stall (alive but no progress) and a sparse heartbeat
set -u

RUN=/e/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64
DIAG=$RUN/diagnostics
CKPT=$RUN/checkpoints
SELFPLAY=$RUN/selfplay
DUMPS=$DIAG/crashdumps
SUPLOG=$DIAG/supervisor.log
HALT=$DIAG/supervisor_halted.flag
DONE=$DIAG/supervisor_completed.flag
EVENTS=$DIAG/events.jsonl
SIG='Fatal Python error|Current thread|panicked|stack backtrace|SIGSEGV|SIGABRT|access violation|0xc0000005|fatal exception|Traceback|STATUS_STACK|STATUS_ACCESS'
SUP_EVENTS='EXIT|RELAUNCH|LAUNCH|CAPTURE|HALT|COMPLETED|ADOPT|WARN|ABORT'

POLL=30
STALL_SECS=1500   # 25 min with no progress while not halted/done
HB_SECS=1800

faults() { grep -Eh "$SIG" $DIAG/trainer.*.err.log 2>/dev/null | wc -l | tr -d ' '; }
ndumps() { ls -1 "$DUMPS"/*.dmp 2>/dev/null | wc -l | tr -d ' '; }
newest_ckpt() { ls -1t "$CKPT"/epoch_*.pt 2>/dev/null | head -1; }
suplines() { [ -f "$SUPLOG" ] && wc -l < "$SUPLOG" | tr -d ' ' || echo 0; }
progress_mtime() {
  { stat -c %Y "$EVENTS" 2>/dev/null
    ls -1t "$SELFPLAY"/*.npz 2>/dev/null | head -1 | xargs -r stat -c %Y 2>/dev/null
    ls -1t "$CKPT"/epoch_*.pt 2>/dev/null | head -1 | xargs -r stat -c %Y 2>/dev/null
  } | sort -nr | head -1
}
emit_eval() {  # $1 = path to dense_cnn.evaluation.epoch_NNNNNN.json
  local f=$1 ep w l c mt st
  ep=$(grep -oE '"epoch":[[:space:]]*[0-9]+' "$f" | grep -oE '[0-9]+' | head -1)
  st=$(grep -oE '"status":[[:space:]]*"[^"]+"' "$f" | sed -E 's/.*"([^"]+)"/\1/' | head -1)
  w=$(grep -oE '"wins":[[:space:]]*[0-9]+' "$f" | grep -oE '[0-9]+' | head -1)
  l=$(grep -oE '"losses":[[:space:]]*[0-9]+' "$f" | grep -oE '[0-9]+' | head -1)
  c=$(grep -oE '"completed":[[:space:]]*[0-9]+' "$f" | grep -oE '[0-9]+' | head -1)
  mt=$(grep -oE '"mean_turns":[[:space:]]*[0-9.]+' "$f" | grep -oE '[0-9.]+' | head -1)
  if [ "$st" = "completed" ] && [ -n "${w:-}" ]; then
    echo "[$(date +%H:%M:%S)] SEALBOT eval epoch ${ep}: ${w} wins / ${l} losses (of ${c}), mean_turns=${mt}  [goal: beat best-50ms]"
  else
    echo "[$(date +%H:%M:%S)] SEALBOT eval epoch ${ep}: status=${st:-?}"
  fi
}

f0=$(faults); d0=$(ndumps); c0=$(newest_ckpt); sl0=$(suplines)
seen_eval=" "
for f in $(ls -1 "$DIAG"/dense_cnn.evaluation.epoch_*.json 2>/dev/null); do
  ep=$(echo "$f" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+'); seen_eval="$seen_eval$ep "
done
halt_latched=0; done_latched=0; stall_latched=0
start=$(date +%s); last_hb=$start
echo "[$(date +%H:%M:%S)] autonomy monitor armed: supervisor-managed run; relaunch gaps are normal. ckpt=$(basename "${c0:-none}") faults=$f0 dumps=$d0"

while true; do
  # halt flag (circuit breaker) -- highest priority
  if [ -f "$HALT" ] && [ "$halt_latched" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] *** CIRCUIT BREAKER HALT *** supervisor stopped relaunching:"
    sed 's/^/    /' "$HALT"
    halt_latched=1
  fi
  if [ -f "$DONE" ] && [ "$done_latched" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] RUN COMPLETED (all configured epochs):"
    sed 's/^/    /' "$DONE"
    done_latched=1
  fi

  # new fault text (rolling across all err logs)
  f1=$(faults)
  if [ "$f1" -gt "$f0" ]; then
    echo "[$(date +%H:%M:%S)] NEW FAULT TEXT in err.log ($f0->$f1):"
    grep -Eh "$SIG" $DIAG/trainer.*.err.log 2>/dev/null | tail -6
    f0=$f1
  fi

  # new minidump
  d1=$(ndumps)
  if [ "$d1" -gt "$d0" ]; then
    echo "[$(date +%H:%M:%S)] NEW MINIDUMP ($d0->$d1): $(ls -1t "$DUMPS"/*.dmp 2>/dev/null | head -1)  -> root-cause from dump+logs"
    d0=$d1
  fi

  # supervisor lifecycle (relaunch/exit/capture/halt/completed)
  sl1=$(suplines)
  if [ "$sl1" -gt "$sl0" ]; then
    tail -n +"$((sl0+1))" "$SUPLOG" 2>/dev/null | grep -E "$SUP_EVENTS" | while read -r ln; do echo "[supervisor] $ln"; done
    sl0=$sl1
  fi

  # new SealBot eval results
  for f in $(ls -1 "$DIAG"/dense_cnn.evaluation.epoch_*.json 2>/dev/null); do
    ep=$(echo "$f" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+')
    case "$seen_eval" in *" $ep "*) : ;; *) emit_eval "$f"; seen_eval="$seen_eval$ep ";; esac
  done

  # progress: new checkpoint
  c1=$(newest_ckpt)
  if [ "$c1" != "$c0" ]; then
    echo "[$(date +%H:%M:%S)] PROGRESS: new checkpoint $(basename "$c1") (was $(basename "${c0:-none}"))"
    c0=$c1
  fi

  # stall (only when not halted/done): alive but no progress for a long time
  now=$(date +%s); pm=$(progress_mtime)
  if [ -z "$(ls -1 "$HALT" "$DONE" 2>/dev/null)" ] && [ -n "${pm:-}" ]; then
    gap=$((now - pm))
    if [ "$gap" -gt "$STALL_SECS" ] && [ "$stall_latched" -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] STALL? no progress (shard/ckpt/events) for ${gap}s (>${STALL_SECS}s) and no relaunch/halt. Possible hang."
      stall_latched=1
    elif [ "$gap" -le "$STALL_SECS" ] && [ "$stall_latched" -eq 1 ]; then
      stall_latched=0
    fi
  fi

  if [ $((now - last_hb)) -ge "$HB_SECS" ]; then
    echo "[$(date +%H:%M:%S)] heartbeat: ckpt=$(basename "${c0:-none}") faults=$f0 dumps=$d0 last_progress=${gap:-?}s ago halt=${halt_latched} done=${done_latched}"
    last_hb=$now
  fi
  sleep "$POLL"
done
