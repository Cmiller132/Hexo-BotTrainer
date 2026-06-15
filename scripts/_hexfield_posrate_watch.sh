#!/bin/bash
# Settle watcher: track epoch-33 CUMULATIVE pos/s as it stabilizes toward the
# epoch average. Reports every 240s; exits when the epoch advances past 33 (true
# average is then the last ep33 reading) or on a fatal error. Pure Git Bash.
RUN=/e/Hexo-BotTrainer/runs/hexfield_main_1
LIVE="$RUN/diagnostics/hexfield.selfplay.live.json"
field(){ grep -oE "\"$1\": [0-9.]+" "$LIVE" 2>/dev/null | head -1 | grep -oE "[0-9.]+$"; }
for i in $(seq 1 12); do
  log=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
  if [ -n "$log" ] && grep -qE "Traceback|CUDA out of memory|RuntimeError|AssertionError" "$log"; then
    echo "ERROR in $log:"; grep -E "Traceback|CUDA out of memory|RuntimeError|AssertionError" "$log" | tail -3
    exit 1
  fi
  ep=$(field epoch); ps=$(field positions_per_second); ac=$(field active_games)
  gf=$(field games_finished); rq=$(field requested_games); el=$(field elapsed_seconds)
  echo "ep$ep cumulative pos/s=$ps active=$ac done=$gf/$rq elapsed=${el}s"
  case "$ep" in 3[4-9]|[4-9][0-9]) echo "ep33 complete (now ep$ep) — last ep33 reading above is the epoch average"; exit 0;; esac
  sleep 240
done
echo "watch window ended (epoch $ep, cumulative pos/s=$(field positions_per_second))"
