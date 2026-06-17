#!/usr/bin/env bash
# Quieter event monitor for the 96x8 run (Monitor tool: each echo = one notification).
# Emits: NEWCKPT (epoch checkpoint landed, with free-RAM readout), SUPER (EXIT/RELAUNCH/
# HALT/COMPLETED, deduped), RAMWARN (watchdog low-RAM, deduped), ALERT (sustained pid
# death), and a sparse HEALTH heartbeat (~every 30 min). Never breaks on a single log
# line. Crash/RAM/checkpoint events are the signal; routine heartbeats are minimized.
# Precise uppercase tokens avoid the false HALT from the lowercase policy banner and the
# "breaker state: ... crashesLastHour=N" heartbeat.
RUN=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8
SUP="$RUN/diagnostics/supervisor_wsl.log"
EVENTS='EXIT pid=[0-9]+ code=[0-9]+|RELAUNCH|HALT|COMPLETED through'
prev_ckpt=""
dead=0
ITERS=20            # 20 * 180s = ~60 min (Monitor timeout cap)
# Prime de-dup so only events that occur DURING the watch fire.
prev_sup=$(tail -8 "$SUP" 2>/dev/null | grep -E "$EVENTS" | tail -1)
prev_ram=$(grep "free RAM" "$SUP" 2>/dev/null | tail -1)
for i in $(seq 1 $ITERS); do
  ckpt=$(ls "$RUN"/checkpoints/epoch_0*.pt 2>/dev/null | sort | tail -1 | grep -oE '[0-9]{6}' | tail -1)
  spe=$(ls -t "$RUN"/selfplay 2>/dev/null | grep -m1 game | grep -oE 'epoch_0*[0-9]+' | head -1)
  pid=$(pgrep -f train_model | head -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  ts=$(date +%H:%M)
  sup=$(tail -8 "$SUP" 2>/dev/null | grep -E "$EVENTS" | tail -1)
  if [ -n "$sup" ] && [ "$sup" != "$prev_sup" ]; then echo "SUPER $ts :: $sup"; prev_sup="$sup"; fi
  ram=$(grep "free RAM" "$SUP" 2>/dev/null | tail -1)
  if [ -n "$ram" ] && [ "$ram" != "$prev_ram" ]; then echo "RAMWARN $ts :: $ram"; prev_ram="$ram"; fi
  if [ -n "$ckpt" ] && [ -n "$prev_ckpt" ] && [ "$ckpt" != "$prev_ckpt" ]; then
    echo "NEWCKPT $ts epoch_$ckpt landed  selfplay=$spe  gpu=[$gpu]  ram_avail=$(free -g | awk '/Mem:/{print $7}')Gi"
  fi
  prev_ckpt="$ckpt"
  if [ -z "$pid" ]; then
    dead=$((dead+1))
    [ "$dead" -ge 2 ] && echo "ALERT $ts trainer pid DEAD ${dead}x consecutive  last_ckpt=$ckpt"
  else
    dead=0
  fi
  if [ $((i % 10)) -eq 1 ]; then
    echo "HEALTH $ts ckpt=$ckpt selfplay=$spe gpu=[$gpu] pid=${pid:-DEAD}"
  fi
  [ "$i" -lt $ITERS ] && sleep 180
done
echo "DONE $(date +%H:%M) watch finished  last_ckpt=$prev_ckpt"
