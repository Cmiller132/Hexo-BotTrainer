#!/usr/bin/env bash
# Watch the 96x8 run until epoch_000010.pt lands (or a halt), polling with sleep.
# Each echo is a Monitor notification. Emits:
#   SPSTART  - self-play epoch advanced (early confirmation each epoch is generating)
#   NEWCKPT  - an epoch checkpoint landed, with free RAM + the cumulative D6-coverage
#              guard hit count (live proof the partial-spill fix is firing)
#   SUPER    - supervisor EXIT/RELAUNCH/HALT/COMPLETED (deduped)
#   RAMWARN  - watchdog low-RAM line (deduped)
#   ALERT    - trainer pid dead >=2 consecutive polls
#   HEALTH   - heartbeat (~every 20 min)
# then REACHED_EPOCH_10 (success) or HALTED, and exits.
RUN=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8
SUP="$RUN/diagnostics/supervisor_wsl.log"
EVENTS='EXIT pid=[0-9]+ code=[0-9]+|RELAUNCH|HALT|COMPLETED through'
prev_ckpt=$(ls "$RUN"/checkpoints/epoch_0*.pt 2>/dev/null | sort | tail -1 | grep -oE '[0-9]{6}' | tail -1)
prev_spe=""
prev_sup=$(tail -8 "$SUP" 2>/dev/null | grep -E "$EVENTS" | tail -1)
prev_ram=$(grep "free RAM" "$SUP" 2>/dev/null | tail -1)
dead=0
i=0
echo "WAIT_START $(date +%H:%M) from ckpt=${prev_ckpt:-none}  target=epoch_000010.pt"
while :; do
  i=$((i + 1))
  ckpt=$(ls "$RUN"/checkpoints/epoch_0*.pt 2>/dev/null | sort | tail -1 | grep -oE '[0-9]{6}' | tail -1)
  spe=$(ls -t "$RUN"/selfplay 2>/dev/null | grep -m1 game | grep -oE 'epoch_0*[0-9]+' | head -1)
  pid=$(pgrep -f train_model | head -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  ram=$(free -g | awk '/Mem:/{print $7}')
  ts=$(date +%H:%M)
  sup=$(tail -8 "$SUP" 2>/dev/null | grep -E "$EVENTS" | tail -1)
  if [ -n "$sup" ] && [ "$sup" != "$prev_sup" ]; then echo "SUPER $ts :: $sup"; prev_sup="$sup"; fi
  rw=$(grep "free RAM" "$SUP" 2>/dev/null | tail -1)
  if [ -n "$rw" ] && [ "$rw" != "$prev_ram" ]; then echo "RAMWARN $ts :: $rw"; prev_ram="$rw"; fi
  if [ -n "$spe" ] && [ "$spe" != "$prev_spe" ]; then echo "SPSTART $ts self-play $spe  gpu=[$gpu] ram=${ram}Gi"; prev_spe="$spe"; fi
  if [ -n "$ckpt" ] && [ "$ckpt" != "$prev_ckpt" ]; then
    guard=$(grep -c "D6 coverage guard" "$RUN"/diagnostics/trainer.*.err.log 2>/dev/null | awk -F: '{s += $2} END {print s + 0}')
    echo "NEWCKPT $ts epoch_$ckpt landed  gpu=[$gpu] ram=${ram}Gi coverage_guard_hits=$guard"
    prev_ckpt="$ckpt"
  fi
  if [ -z "$pid" ]; then
    dead=$((dead + 1))
    [ "$dead" -ge 2 ] && echo "ALERT $ts trainer pid DEAD ${dead}x consecutive  ckpt=$ckpt"
  else
    dead=0
  fi
  if [ $((i % 10)) -eq 1 ]; then echo "HEALTH $ts ckpt=$ckpt selfplay=$spe gpu=[$gpu] ram=${ram}Gi pid=${pid:-DEAD}"; fi
  if [ -f "$RUN"/checkpoints/epoch_000010.pt ]; then
    echo "REACHED_EPOCH_10 $ts ckpt=$ckpt gpu=[$gpu] ram=${ram}Gi -- run healthy through epoch 10"
    break
  fi
  if [ -f "$RUN"/diagnostics/supervisor_halted.flag ]; then
    echo "HALTED $ts :: $(tail -1 "$RUN"/diagnostics/supervisor_halted.flag 2>/dev/null)"
    break
  fi
  sleep 120
done
echo "WAIT_DONE $(date +%H:%M)"
