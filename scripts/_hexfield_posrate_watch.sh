#!/bin/bash
# Quiet delta-watcher: poll every 180s but ONLY emit on a meaningful change —
# epoch advance (epoch-33 final average appears when ep becomes 34), a fatal
# error, or a >15% pos/s move from the last emitted value. Exits at epoch >=34
# or on error. Pure Git Bash.
RUN=/e/Hexo-BotTrainer/runs/hexfield_main_1
LIVE="$RUN/diagnostics/hexfield.selfplay.live.json"
field(){ grep -oE "\"$1\": [0-9.]+" "$LIVE" 2>/dev/null | head -1 | grep -oE "[0-9.]+$"; }
last_ep=""; last_ps=""
emit(){ echo "$1"; }
for i in $(seq 1 60); do
  log=$(ls -t "$RUN"/train.*.out.log 2>/dev/null | head -1)
  if [ -n "$log" ] && grep -qE "Traceback|CUDA out of memory|RuntimeError|AssertionError" "$log"; then
    emit "ERROR in $log:"; grep -E "Traceback|CUDA out of memory|RuntimeError|AssertionError" "$log" | tail -3
    exit 1
  fi
  ep=$(field epoch); ps=$(field positions_per_second); gf=$(field games_finished); rq=$(field requested_games)
  if [ -n "$ep" ]; then
    if [ "$ep" != "$last_ep" ]; then
      [ -n "$last_ep" ] && emit "EPOCH $last_ep DONE -> now ep$ep. (ep$last_ep final cumulative pos/s ~= last reading)"
      last_ep="$ep"; last_ps="$ps"
      case "$ep" in 3[4-9]|[4-9][0-9]) emit "epoch 34 reached — ep33 (active_games=96) is complete; final average was the last ep33 reading."; exit 0;; esac
      emit "ep$ep pos/s=$ps done=$gf/$rq"
    elif [ -n "$ps" ] && [ -n "$last_ps" ]; then
      d=$(awk -v a="$ps" -v b="$last_ps" 'BEGIN{if(b>0){r=(a-b)/b; if(r<0)r=-r; print (r>0.15)?1:0}else print 0}')
      if [ "$d" = "1" ]; then emit "ep$ep pos/s moved $last_ps -> $ps (>15%) done=$gf/$rq"; last_ps="$ps"; fi
    fi
  fi
  sleep 180
done
emit "watch window ended (ep$last_ep pos/s=$(field positions_per_second))"
