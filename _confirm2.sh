#!/bin/bash
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== now ==="; date +%H:%M:%S
echo "=== config header + all epoch/eval lines so far ==="
head -2 $R/rl_train.log 2>/dev/null
grep -E "epoch [0-9]+ (selfplay|train|done)|EVAL|BASELINE" $R/rl_train.log 2>/dev/null
echo "=== epoch-0 selfplay progress (/256) ==="
ls $R/selfplay/epoch_000000_game_*.npz 2>/dev/null | wc -l
echo "=== supervisor + driver + bridge alive? ==="
for p in 2160571 2160600 2160693; do ps -o pid= -p $p >/dev/null 2>&1 && echo "pid $p ALIVE ($(ps -o etimes= -p $p|tr -d ' ')s)" || echo "pid $p DEAD"; done
echo "=== RAM + GPU ==="
free -g | grep Mem
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
echo "=== relaunches? ==="
grep -c LAUNCH $R/supervisor.log 2>/dev/null
