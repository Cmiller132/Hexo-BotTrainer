#!/usr/bin/env bash
cd /mnt/e/Hexo-BotTrainer-hexgt
echo "############ 1. DRIVER PROCESS (pid + start time + elapsed) ############"
for p in $(pgrep -f _rl_train.py); do
  ps -o pid,lstart,etimes,cmd -p "$p" | tail -1 | cut -c1-160
done
echo "now: $(date)"
echo
echo "############ 2. selfplay.py edit time vs driver start ############"
stat -c '%y  %n' packages/hexo_models/hexgt/python/hexo_models/hexgt/selfplay.py
echo "_write_game_record present in source? -> $(grep -c _write_game_record packages/hexo_models/hexgt/python/hexo_models/hexgt/selfplay.py)"
echo
echo "############ 3. CHECKPOINTS (latest epoch) ############"
ls -la --time-style=+%H:%M:%S runs/hexgt_rl_main2/checkpoints/*.pt 2>/dev/null | tail -8
echo
echo "############ 4. supervisor.log relaunch history ############"
grep -E "LAUNCH|RELAUNCH|RESUME|EXIT|SUPERVISOR start" runs/hexgt_rl_main2/supervisor.log 2>/dev/null | tail -12
echo
echo "############ 5. rl_train.log RL-start lines ############"
grep -E "RL start|RESUME|SEED" runs/hexgt_rl_main2/rl_train.log 2>/dev/null | tail -8
echo
echo "############ 6. REAL .hxr in worktree selfplay/ (driver writes here) ############"
ls -la --time-style=+%H:%M:%S runs/hexgt_rl_main2/selfplay/*.hxr 2>/dev/null
echo "current selfplay shard epochs:"
ls runs/hexgt_rl_main2/selfplay/*.npz 2>/dev/null | sed -E 's/.*epoch_([0-9]+)_game.*/\1/' | sort | uniq -c | tail -6
