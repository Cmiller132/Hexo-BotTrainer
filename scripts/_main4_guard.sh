#!/usr/bin/env bash
# main_4 overnight guard: failure-only watcher. Polls every 15 min for 8 h.
# Exit 1 on supervisor death / traceback / vf>0 (with evidence); exit 0 at the
# end of the shift with a latest-epoch summary.
set -u
RUN=/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4
D=$RUN/diagnostics
for i in $(seq 1 32); do
  sleep 900
  if ! pgrep -f "[s]upervise_main1" >/dev/null; then
    echo "FAIL: supervisor gone"; tail -10 "$RUN/supervisor.log"; exit 1
  fi
  L=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
  if [ -n "${L:-}" ] && grep -qi "traceback" "$L"; then
    echo "FAIL: traceback"; grep -iA5 traceback "$L" | head -30; exit 1
  fi
  J=$(ls -t "$D"/hexfield.selfplay.epoch_*.json 2>/dev/null | head -1)
  if [ -n "${J:-}" ]; then
    vf=$(python3 -c "import json,sys;d=json.load(open('$J'));print(d.get('tss',d).get('deep_verify_failed'))" 2>/dev/null)
    if [ "${vf:-0}" != "0" ] && [ "${vf:-0}" != "None" ]; then
      echo "FAIL: deep_verify_failed=$vf in $J"; exit 1
    fi
  fi
done
echo "shift complete; latest epoch:"
J=$(ls -t "$D"/hexfield.selfplay.epoch_*.json 2>/dev/null | head -1)
echo "$J"
python3 -c "import json;d=json.load(open('$J'));t=d.get('tss',d);p=t.get('park_parked',0) or 1;print({'vf':t.get('deep_verify_failed'),'win':t.get('deep_win'),'loss':t.get('deep_loss'),'bail_pct':round(100.0*t.get('park_bailed',0)/p,1)})" 2>/dev/null
exit 0
