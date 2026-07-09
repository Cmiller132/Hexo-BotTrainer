#!/usr/bin/env bash
# One-shot clean slate for the hexfield systemd services: stop+disable both
# units, kill every stray supervisor/trainer/dashboard process (excluding this
# script + its parent), clear the stale lock, and report remaining strays.
# Run as a FILE (wsl.exe -e bash <path>) to avoid nested -c quoting issues.
set -u
# Defaults target the main_9 supervisor unit/run; override UNIT/RUN for the
# main_11 or eq-1 units (e.g. UNIT=hexfield-supervisor-11
# RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_main_11).
UNIT="${UNIT:-hexfield-supervisor-9}"
RUN="${RUN:-/mnt/e/Hexo-BotTrainer/runs/hexfield_main_9}"
self=$$; parent=$PPID

echo "[reset] stop + disable units"
systemctl stop "$UNIT" hexfield-dashboard 2>/dev/null
systemctl disable "$UNIT" hexfield-dashboard 2>/dev/null
systemctl reset-failed "$UNIT" hexfield-dashboard 2>/dev/null
sleep 2

scan_and_kill() {
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    [ "$pid" = "$self" ] && continue
    [ "$pid" = "$parent" ] && continue
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    case "$cmd" in
      *_hexfield_supervise_main1.sh*|*hexo_train.cli.train_model*|*hexo_frontend.web*)
        echo "  kill $pid : $(printf '%.70s' "$cmd")"
        kill -9 "$pid" 2>/dev/null ;;
    esac
  done
}

echo "[reset] killing stray processes (pass 1)"
scan_and_kill
sleep 2
echo "[reset] killing stray processes (pass 2)"
scan_and_kill
sleep 1
rm -f "$RUN/supervisor.lock"

# count survivors
cnt=0
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ "$pid" = "$self" ] && continue
  [ "$pid" = "$parent" ] && continue
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
  case "$cmd" in
    *_hexfield_supervise_main1.sh*|*hexo_train.cli.train_model*|*hexo_frontend.web*) cnt=$((cnt+1)) ;;
  esac
done
echo "[reset] remaining stray_count=$cnt"
echo "[reset] supervisor unit ($UNIT): $(systemctl is-active "$UNIT" 2>/dev/null) / $(systemctl is-enabled "$UNIT" 2>/dev/null)"
echo "[reset] GPU mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
echo "[reset] done at $(date -u +%H:%M:%SZ)"
