#!/usr/bin/env bash
# Real-time crash/stall watcher for the dense_cnn scratch_64 trainer.
# Emits one stdout line per notable event (Monitor turns each into a chat
# notification). NO auto-relaunch. On death it FREEZES artifacts into a
# timestamped crash_artifacts folder before exiting, so a backstop relaunch
# cannot clobber the evidence.
set -u

PID=${1:-10672}
RUN=/e/Hexo-BotTrainer/runs/dense_cnn_model1_scratch_64
DIAG=$RUN/diagnostics
ERR=${2:-$DIAG/trainer.20260528_231854.err.log}
OUT=${OUT:-$DIAG/trainer.20260528_231854.out.log}
EVENTS=$DIAG/events.jsonl
WATCHDOG=$DIAG/resource_watchdog.jsonl
DUMPS=$DIAG/crashdumps
CKPT=$RUN/checkpoints
SELFPLAY=$RUN/selfplay
SIG='Fatal Python error|Current thread|panicked|stack backtrace|SIGSEGV|SIGABRT|access violation|0xc0000005|fatal exception|Traceback|STATUS_STACK|STATUS_ACCESS'

POLL=30            # seconds between polls
STALL_SECS=1500    # 25 min with zero progress while alive => stall (epoch ~16 min)
HB_SECS=1800       # heartbeat every 30 min

alive() { MSYS_NO_PATHCONV=1 tasklist /FI "PID eq $PID" /NH 2>/dev/null | grep -qi 'python\.exe'; }
faults() { grep -Ec "$SIG" "$ERR" 2>/dev/null || echo 0; }
ndumps() { ls -1 "$DUMPS"/*.dmp 2>/dev/null | wc -l | tr -d ' '; }
newest_ckpt() { ls -1t "$CKPT"/epoch_*.pt 2>/dev/null | head -1; }
# newest progress mtime (epoch secs) across selfplay shards + checkpoints + events
progress_mtime() {
  { stat -c %Y "$EVENTS" 2>/dev/null
    ls -1t "$SELFPLAY"/*.npz 2>/dev/null | head -1 | xargs -r stat -c %Y 2>/dev/null
    ls -1t "$CKPT"/epoch_*.pt 2>/dev/null | head -1 | xargs -r stat -c %Y 2>/dev/null
  } | sort -nr | head -1
}

f0=$(faults); d0=$(ndumps); c0=$(newest_ckpt); start=$(date +%s)
last_hb=$start; stall_latched=0
echo "[$(date +%H:%M:%S)] monitor armed: PID=$PID err=$(basename "$ERR") faults=$f0 dumps=$d0 ckpt=$(basename "${c0:-none}")"

capture_and_exit() {
  local reason=$1
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local AD=$DIAG/crash_artifacts/$stamp
  mkdir -p "$AD"
  echo "[$(date +%H:%M:%S)] *** $reason *** freezing artifacts -> crash_artifacts/$stamp"
  # Give WER a chance to finish writing a fresh full dump (up to 120s, wait for size to stabilize).
  if [ "$reason" = "PROCESS DEAD" ]; then
    local waited=0 prev=-1 cur
    while [ $waited -lt 120 ]; do
      cur=$(ls -1 "$DUMPS"/*.dmp 2>/dev/null | wc -l)
      if [ "$cur" -gt "$d0" ]; then
        local sz; sz=$(ls -1t "$DUMPS"/*.dmp 2>/dev/null | head -1 | xargs -r stat -c %s 2>/dev/null)
        [ "$sz" = "$prev" ] && break   # size stable => dump complete
        prev=$sz
      fi
      sleep 6; waited=$((waited+6))
    done
  fi
  cp "$ERR" "$AD/" 2>/dev/null
  cp "$OUT" "$AD/" 2>/dev/null
  tail -80 "$EVENTS"   > "$AD/events.tail.jsonl"   2>/dev/null
  tail -30 "$WATCHDOG" > "$AD/watchdog.tail.jsonl" 2>/dev/null
  cp "$DUMPS"/*.dmp "$AD/" 2>/dev/null
  local sigtxt; sigtxt=$(grep -E "$SIG" "$ERR" 2>/dev/null | tail -8)
  local dmp; dmp=$(ls -1t "$AD"/*.dmp 2>/dev/null | head -1)
  {
    echo "## $stamp — $reason (PID $PID)"
    echo "- err.log: $ERR"
    echo "- artifacts: $AD"
    echo "- minidump: ${dmp:-NONE (WER LocalDumps not configured, or no faulting-module crash)}"
    echo "- fault signature (tail of err.log):"
    if [ -n "$sigtxt" ]; then echo '```'; echo "$sigtxt"; echo '```'
    else echo "  (no PYTHONFAULTHANDLER / panic / traceback text in err.log — silent native exit)"; fi
    echo "- last event:"
    echo '```'; tail -1 "$EVENTS" 2>/dev/null; echo '```'
    echo
  } >> "$DIAG/crashlog.md"
  echo "[$(date +%H:%M:%S)] artifacts frozen. dump=${dmp:-none}. signature: ${sigtxt:-<none — silent native exit>}"
  echo "[$(date +%H:%M:%S)] crashlog.md appended. monitor exiting — ROOT-CAUSE NOW, no relaunch."
  exit 0
}

while true; do
  if ! alive; then capture_and_exit "PROCESS DEAD"; fi

  f1=$(faults)
  if [ "$f1" -gt "$f0" ]; then
    echo "[$(date +%H:%M:%S)] NEW FAULT TEXT in err.log (count $f0->$f1):"
    grep -E "$SIG" "$ERR" 2>/dev/null | tail -6
    f0=$f1
  fi

  d1=$(ndumps)
  if [ "$d1" -gt "$d0" ]; then
    echo "[$(date +%H:%M:%S)] NEW MINIDUMP in crashdumps/ ($d0->$d1): $(ls -1t "$DUMPS"/*.dmp 2>/dev/null | head -1)"
    d0=$d1
  fi

  c1=$(newest_ckpt)
  if [ "$c1" != "$c0" ]; then
    echo "[$(date +%H:%M:%S)] PROGRESS: new checkpoint $(basename "$c1") (was $(basename "${c0:-none}"))"
    c0=$c1
  fi

  now=$(date +%s); pm=$(progress_mtime)
  if [ -n "${pm:-}" ]; then
    gap=$((now - pm))
    if [ "$gap" -gt "$STALL_SECS" ] && [ "$stall_latched" -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] STALL? alive but no progress (shard/ckpt/events) for ${gap}s (>${STALL_SECS}s). Investigate hang."
      stall_latched=1
    elif [ "$gap" -le "$STALL_SECS" ] && [ "$stall_latched" -eq 1 ]; then
      echo "[$(date +%H:%M:%S)] progress resumed (gap ${gap}s); stall cleared."
      stall_latched=0
    fi
  fi

  if [ $((now - last_hb)) -ge "$HB_SECS" ]; then
    echo "[$(date +%H:%M:%S)] heartbeat: alive, ckpt=$(basename "${c0:-none}"), faults=$f0, dumps=$d0, last_progress=${gap:-?}s ago"
    last_hb=$now
  fi

  sleep "$POLL"
done
