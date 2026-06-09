#!/bin/bash
R=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2
echo "=== now ==="; date +%H:%M:%S
echo "=== seed + config header ==="; head -2 $R/rl_train.log 2>/dev/null
echo "=== baseline + epoch/eval lines ==="; grep -E "BASELINE|EVAL|epoch [0-9]+ (selfplay|train|done)" $R/rl_train.log 2>/dev/null
echo "=== epoch-0 selfplay progress (/256) ==="; ls $R/selfplay/epoch_000000_game_*.npz 2>/dev/null | wc -l
echo "=== checkpoints + latest mtime ==="; ls -1t --time-style=+%H:%M:%S $R/checkpoints/hexgt_rl_epoch*.pt 2>/dev/null | head -1; echo "count: $(ls $R/checkpoints/hexgt_rl_epoch*.pt 2>/dev/null | wc -l)"
echo "=== procs: supervisor/driver/bridge + relaunches ==="
ps -eo pid,etimes,comm | grep -E "python|bash" | grep -v grep >/dev/null
for pat in "_rl_supervise" "_rl_train.py" "_dashboard_bridge"; do echo "$pat: $(ps -eo pid,etimes,args|grep -E "$pat"|grep -v grep|head -1|awk '{print $1" ("$2"s)"}')"; done
echo "relaunch count: $(grep -c LAUNCH $R/supervisor.log 2>/dev/null)  halt? $(ls $R/supervisor_halted.flag 2>/dev/null && echo YES || echo no)"
echo "=== GPU (2 samples) + RAM ==="; for i in 1 2; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null; sleep 1; done; free -g | grep Mem
