#!/usr/bin/env bash
# Periodic health snapshots for the 96x8 run, appended to health_watch.log
# (tailed by a Monitor). Runs in WSL so nvidia-smi / /proc / pgrep are real.
RUN=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8
LOG="$RUN/diagnostics/health_watch.log"
for i in $(seq 1 6); do
  le=$(ls "$RUN"/checkpoints/epoch_0*.pt 2>/dev/null | sort | tail -1 | grep -oE "[0-9][0-9]+" | tail -1)
  spc=$(ls "$RUN"/selfplay 2>/dev/null | grep -c game)
  spe=$(ls -t "$RUN"/selfplay 2>/dev/null | grep game | head -1 | grep -oE "epoch_0*[0-9]+" | head -1)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  ramkb=$(grep MemAvailable /proc/meminfo | grep -oE "[0-9]+" | head -1)
  ramgb=$((ramkb / 1048576))
  pid=$(pgrep -f train_model | head -1)
  alert=$(tail -8 "$RUN"/diagnostics/supervisor_wsl.log 2>/dev/null | grep -E "code=[1-9]|HALT|consecFast=[1-9]" | tail -1)
  echo "HEALTH $(date +%H:%M) ckpt_epoch=${le:-none} selfplay=${spe:-NA}/${spc} gpu=[${gpu}] ram_gb=${ramgb} pid=${pid:-DEAD} ${alert:+!!ALERT=$alert}" >> "$LOG"
  [ "$i" -lt 6 ] && sleep 570
done
echo "HOUR_WATCH_DONE $(date +%H:%M)" >> "$LOG"
