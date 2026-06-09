#!/bin/bash
echo "=== RL process tree (supervisor 2140517 + driver) ==="
ps -eo pid,ppid,pgid,sid,args | grep -E "_rl_supervise|_rl_train.py|_rl_run_fg" | grep -v grep | cut -c1-130
echo "=== driver pid currently (from pidfile) ==="
cat /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/driver.pid 2>/dev/null
echo "=== current epoch ==="
tail -2 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/rl_train.log
