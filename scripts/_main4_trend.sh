#!/usr/bin/env bash
# main_4 epoch 2-4 trend watcher: park-bail trajectory + soundness invariants.
# Exit 0 after epoch 4 summary; exit 1 on vf>0 / supervisor death / traceback.
set -u
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4
D=$RUN/diagnostics
last=1
for i in $(seq 1 48); do
  sleep 300
  if ! pgrep -f "[s]upervise_main1" >/dev/null; then
    echo "FAIL: supervisor gone"; tail -10 "$RUN/supervisor.log"; exit 1
  fi
  L=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
  if [ -n "${L:-}" ] && grep -qi "traceback" "$L"; then
    echo "FAIL: traceback"; grep -iA5 traceback "$L" | head -30; exit 1
  fi
  for ep in 2 3 4; do
    J=$D/hexfield.selfplay.epoch_00000${ep}.json
    if [ "$ep" -gt "$last" ] && [ -f "$J" ]; then
      last=$ep
      python3 - "$J" "$ep" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); t = d.get("tss", d)
p = t.get("park_parked", 0) or 1
print(f"epoch {sys.argv[2]}: vf={t.get('deep_verify_failed')} "
      f"win={t.get('deep_win')} loss={t.get('deep_loss')} "
      f"bail={100.0*t.get('park_bailed',0)/p:.1f}% "
      f"avg_wait={t.get('park_wait_ms_sum',0)/p:.0f}ms "
      f"wbk={t.get('deep_win_backups')} lbk={t.get('deep_loss_backups')}")
sys.exit(0 if t.get("deep_verify_failed") == 0 else 3)
EOF
      [ $? -ne 0 ] && { echo "FAIL: vf nonzero at epoch $ep"; exit 1; }
    fi
  done
  [ "$last" -ge 4 ] && { echo "trend complete through epoch 4"; exit 0; }
done
echo "TIMEOUT after 4h; last epoch seen: $last"; exit 2
