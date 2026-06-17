#!/usr/bin/env bash
# Confirm the k=2 forced-playout path runs in production: wait for epoch-2 selfplay
# to adopt TRT and start writing shards, or report a crash/HALT loudly. Exits 0 on
# selfplay active, 1 on failure/timeout.
D=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6
for i in $(seq 1 40); do          # ~40*20s = ~13 min
  err=$(grep -hiE 'Traceback|forced_playout_k|panicked|CUDA error|NOT ADOPTED|TRTAdoptError' "$D"/diagnostics/trainer.20260530_025514.err.log 2>/dev/null | tail -3)
  halt=$(ls "$D"/diagnostics/supervisor_halted.flag 2>/dev/null)
  if [ -n "$halt" ]; then echo "HALT FLAG PRESENT"; cat "$halt" 2>/dev/null; exit 1; fi
  if [ -n "$err" ]; then echo "TRAINER ERROR DETECTED:"; echo "$err"; exit 1; fi
  trt=$(grep -h 'adopted TRT FP16' "$D"/diagnostics/trainer.20260530_025514.out.log 2>/dev/null | tail -1)
  shards=$(ls "$D"/selfplay/ 2>/dev/null | grep -c 'epoch_000002_game_.*\.npz')
  if [ -n "$trt" ] && [ "$shards" -ge 1 ]; then
    echo "EPOCH-2 SELFPLAY ACTIVE WITH k=2 (t+$((i*20))s): shards=$shards"
    echo "TRT: $trt"
    exit 0
  fi
  sleep 20
done
echo "TIMEOUT: epoch-2 selfplay did not start"; exit 1
