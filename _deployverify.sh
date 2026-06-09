#!/usr/bin/env bash
echo "=== driver argv (features+replay code) ==="
p=$(pgrep -f _rl_train.py | head -1)
echo "driver pid=$p start=$(ps -o lstart= -p "$p" 2>/dev/null)"
tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE -- "--max-actions [0-9]+|--eval-every [0-9]+|--visits [0-9]+|--active [0-9]+"
echo "widening default in live code:"; grep -E "widening-max-children" /mnt/e/Hexo-BotTrainer-hexgt/scripts/_rl_train.py | grep default
echo "feature slots in live constants:"; grep -E "F_CAND_OWN_WIN3|FEATURE_SCHEMA_VERSION" /mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/hexgt/python/hexo_models/hexgt/constants.py | head -2
echo "=== epoch progress ==="
grep -E "RESUME|ZERO-INIT|epoch [0-9]+ selfplay|epoch [0-9]+ done" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/rl_train.log | tail -2
echo "ep9 games: $(ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_000009_game_*.npz 2>/dev/null | wc -l)/256"
echo "=== resources ==="
awk '/MemAvailable/{printf "RAM free %.1f GB\n", $2/1048576}' /proc/meminfo
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== stop old dashboard (bridge + 8080) so durable relaunch can take over ==="
for bp in $(pgrep -f _dashboard_bridge.py); do echo "kill bridge $bp"; kill "$bp" 2>/dev/null; done
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/_kill8080.sh | bash
