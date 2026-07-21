#!/usr/bin/env bash
# main_4 first-epoch watcher (orchestrator overnight instrument, 2026-07-21).
# Polls every 5 min for up to 2 h. Exit codes: 0 = first epoch JSON landed and
# tss block healthy (prints it); 1 = failure (supervisor gone / traceback /
# crash loop / verify failures); 2 = timeout without an epoch JSON.
set -u
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4
for i in $(seq 1 24); do
  sleep 300
  if ! pgrep -f "[s]upervise_main1" >/dev/null; then
    echo "FAIL: supervisor process gone"
    tail -10 "$RUN/supervisor.log" 2>/dev/null
    exit 1
  fi
  L=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
  if [ -n "${L:-}" ] && grep -qi "traceback" "$L"; then
    echo "FAIL: traceback in $L"
    grep -iA5 "traceback" "$L" | head -30
    exit 1
  fi
  launches=$(grep -c "LAUNCH out=" "$RUN/supervisor.log" 2>/dev/null || echo 0)
  if [ "$launches" -gt 3 ]; then
    echo "FAIL: crash loop ($launches launches)"
    tail -20 "$RUN/supervisor.log"
    exit 1
  fi
  J=$(grep -rls deep_verify_failed "$RUN" --include='*.json' 2>/dev/null | xargs -r ls -t | head -1)
  if [ -n "${J:-}" ]; then
    echo "first epoch JSON: $J"
    python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
tss = d.get("tss", d)
keys = ["deep_verify_failed", "deep_calls", "deep_win", "deep_loss",
        "deep_hard_backups", "deep_win_backups", "deep_loss_backups",
        "park_parked", "park_bailed", "park_wait_ms_sum"]
print({k: tss.get(k) for k in keys})
print("lr:", d.get("lr"))
vf = tss.get("deep_verify_failed", None)
sys.exit(0 if vf == 0 else 3)
EOF
    rc=$?
    [ $rc -eq 0 ] && exit 0
    echo "FAIL: deep_verify_failed nonzero or unreadable (rc=$rc)"
    exit 1
  fi
done
echo "TIMEOUT: no epoch JSON within 2h"
exit 2
