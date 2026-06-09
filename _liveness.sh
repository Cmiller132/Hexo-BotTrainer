#!/bin/bash
echo "=== RL procs ==="
for p in 2140517 2140546; do
  if ps -o pid= -p $p >/dev/null 2>&1; then
    echo "pid $p ALIVE: $(ps -o etimes=,comm= -p $p | tr -s ' ')"
  else echo "pid $p DEAD"; fi
done
echo "=== bridge proc(s) ==="
ps -eo pid,etimes,args | grep _dashboard_bridge | grep -v grep | cut -c1-70
echo "=== main-tree mirror freshness (newest files + does epoch 4 show?) ==="
M=/mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main/diagnostics
ls -1t --time-style=+%H:%M:%S "$M" | head -3
echo "epoch4 selfplay json present? $(ls "$M"/dense_cnn.selfplay.epoch_000004.json 2>/dev/null && echo yes || echo NO)"
echo "live.json mtime: $(stat -c '%y' "$M/dense_cnn.selfplay.live.json" 2>/dev/null | cut -d. -f1)"
echo "bridge log tail:"; tail -2 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/_bridge.log 2>/dev/null
echo "=== eval cadence: which epochs evaled? ==="
grep -oE "epoch [0-9]+ EVAL" /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/rl_train.log 2>/dev/null
