#!/usr/bin/env bash
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
LOG="$RUN/driver.20260604_023004.out.log"
echo "=== full driver log (current) ==="
cat "$LOG"
echo "=== selfplay shards ==="
ls -lt "$RUN/selfplay/" 2>/dev/null | head -6
echo "=== events.jsonl tail ==="
tail -4 "$RUN/events.jsonl" 2>/dev/null || echo "(no events.jsonl)"
echo "=== gpu telemetry tail ==="
tail -4 "$RUN/gpu_telemetry.csv" 2>/dev/null || echo "(no telemetry)"
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "(no smi)"
