#!/usr/bin/env bash
echo "=== driver argv (bundle flags) ==="
p=$(pgrep -f _rl_train.py | head -1)
echo "driver pid=$p start=$(ps -o lstart= -p "$p" 2>/dev/null)"
tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--max-actions [0-9]+|--widening-max-children [0-9]+|--eval-every [0-9]+|--visits [0-9]+|--active [0-9]+"
echo "=== sp.__file__ (NEW code from live worktree) ==="
cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep -E "^PYTHONPATH=" | head -1 | cut -c1-90
echo "=== widening default in running code ==="
grep -A1 'widening-max-children' /mnt/e/Hexo-BotTrainer-hexgt/scripts/_rl_train.py | grep default | head -1
echo "=== epoch-8 self-play + full .hxr (bundle: full game records) ==="
ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_000008_game_*.npz 2>/dev/null | wc -l
ls -la --time-style=+%H:%M /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_000008.hxr 2>/dev/null || echo "(epoch_000008.hxr not created yet — appears once self-play opens it)"
echo "=== dashboard bridge + 8080 ==="
pgrep -af _dashboard_bridge.py | grep -v pgrep | head -1 || echo "bridge DOWN"
curl -s -o /dev/null -w "8080 http=%{http_code}\n" localhost:8080
echo "=== resources ==="
awk '/MemAvailable/{printf "RAM free %.1f / 27.4 GB\n", $2/1048576}' /proc/meminfo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
