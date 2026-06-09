#!/usr/bin/env bash
echo "=== driver start line + loss weights (stv should be 0.1) ==="
grep -E "RESUME|loss_w:" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log | tail -2
echo "=== driver argv has --short-term-value-weight 0.10 ? ==="
p=$(pgrep -f _rl_train.py | head -1)
echo "driver pid=$p"
tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--short-term-value-weight [0-9.]+|--max-actions [0-9]+"
echo "=== supervisor watchdogs (bridge + telemetry now BOTH managed) ==="
grep -E "SUPERVISOR start|bridge watchdog|telemetry sampler watchdog" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/supervisor.log | tail -3
echo "=== managed children ==="
echo "bridge count: $(pgrep -fc _dashboard_bridge.py)  telemetry count: $(pgrep -fc 'query-gpu=timestamp,temperature')"
echo "=== GPU + RAM ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
awk '/MemAvailable/{printf "RAM free %.1f GB\n", $2/1048576}' /proc/meminfo
