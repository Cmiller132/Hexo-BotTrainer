#!/bin/bash
# Stop supervisor FIRST (so it can't relaunch), then the driver. Keep all data.
echo "killing supervisor 2140517..."; kill -TERM 2140517 2>/dev/null
sleep 2
echo "killing driver 2140546..."; kill -TERM 2140546 2>/dev/null
sleep 3
# force any stragglers + the wrapper
kill -9 2140517 2140546 2140506 2>/dev/null
pkill -9 -f "_rl_train.py" 2>/dev/null
pkill -9 -f "_rl_supervise.sh" 2>/dev/null
sleep 2
echo "=== survivors (none = clean) ==="
ps -eo pid,args 2>/dev/null | grep -E "_rl_train.py|_rl_supervise|_rl_run_fg" | grep -v grep | head
echo "=== old run checkpoints preserved? ==="
ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/checkpoints/hexgt_rl_epoch*.pt 2>/dev/null | wc -l
