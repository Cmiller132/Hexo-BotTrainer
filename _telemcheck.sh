#!/usr/bin/env bash
CSV=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/gpu_telemetry.csv
echo "=== sampler process ==="
pgrep -af "query-gpu=timestamp,temperature" | grep -v pgrep | head -2
echo "=== CSV lines + last 4 rows ==="
wc -l < "$CSV"
tail -4 "$CSV"
echo "=== throttle field column (last value) ==="
tail -1 "$CSV" | awk -F', ' '{print "clocks_throttle_reasons.active =", $NF}'
echo "=== run still healthy (driver + bridge) ==="
echo "driver: $(pgrep -f _rl_train.py | head -1)  bridge count: $(pgrep -fc _dashboard_bridge.py)"
