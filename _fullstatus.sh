#!/usr/bin/env bash
echo "=== 1. TRAINING DRIVER (survived 8080 restart?) ==="
p=$(pgrep -f _rl_train.py | head -1)
if [ -n "$p" ]; then
  echo "driver pid=$p  start=$(ps -o lstart= -p "$p")  elapsed=$(ps -o etimes= -p "$p" | tr -d ' ')s  -> ALIVE"
else
  echo "driver NOT running"
fi
echo "supervisor:"; pgrep -af '_rl_supervise|_rl_run_fg' | grep -v pgrep | head -1
echo "latest rl_train.log:"; tail -1 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log
echo "current self-play epoch progress:"; ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/*.npz 2>/dev/null | sed -E 's/.*epoch_([0-9]+)_game.*/\1/' | sort | uniq -c | tail -2
echo; echo "=== 2. DASHBOARDS ==="
curl -s -o /dev/null -w "8080 http=%{http_code}\n" localhost:8080
curl -s -o /dev/null -w "8081 http=%{http_code}\n" localhost:8081 2>/dev/null || echo "8081 DOWN"
echo "bridge:"; pgrep -af _dashboard_bridge.py | grep -v pgrep | head -1
echo; echo "=== 3. RESOURCES ==="
awk '/MemAvailable/{printf "RAM free %.1f / 27.4 GB\n", $2/1048576}' /proc/meminfo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
