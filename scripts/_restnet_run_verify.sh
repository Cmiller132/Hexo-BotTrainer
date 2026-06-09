#!/usr/bin/env bash
# Verify the live dense_cnn_restnet_main1 run: arch, init-from-prefit, visits,
# calibration, self-play producing games, GPU/VRAM, pos/s. Polls ~8 min.
RUNDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
LOG=$(ls -1t "$RUNDIR"/train.*.out.log 2>/dev/null | head -1)
echo "train log: $LOG"
for i in $(seq 1 9); do
  echo "----- poll $i -----"
  echo "[arch/init/visits/calib markers]"
  grep -iE 'blocks_type|R_R_R_T|params|param count|initialize_from|loaded checkpoint|checkpoint_ref|search_visits|visits=|calibrat|selected_|self-play|selfplay|epoch 0|games' "$LOG" 2>/dev/null | tail -8
  echo "[load_checkpoint.json]"
  cat "$RUNDIR/diagnostics/load_checkpoint.json" 2>/dev/null | tr -d '\n' | head -c 400; echo
  echo "[selfplay live]"
  cat "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" 2>/dev/null | tr -d '\n' | head -c 400; echo
  echo "[GPU]"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1
  # stop early once self-play reports games in flight
  if grep -qE '"games"' "$RUNDIR/diagnostics/dense_cnn.selfplay.live.json" 2>/dev/null; then
    echo "=== SELF-PLAY LIVE ==="; break
  fi
  if grep -qiE 'Traceback|out of memory|CUDA error|FATAL|halt' "$LOG" 2>/dev/null; then
    echo "=== ERROR DETECTED ==="; tail -25 "$LOG"; break
  fi
  sleep 55
done
echo "===== final tail ====="
tail -20 "$LOG" 2>/dev/null
