#!/usr/bin/env bash
RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3
echo "=== flags ==="
for f in supervisor_halted.flag supervisor_completed.flag; do
  if [ -e "$RD/$f" ]; then echo "EXISTS $f :: $(cat "$RD/$f")"; else echo "absent  $f"; fi
done
echo "=== latest 2 epoch checkpoints ==="
ls -t "$RD/checkpoints"/hexgt_rl_epoch*.pt 2>/dev/null | head -2
echo "=== latest hexgt_rl_latest.pt mtime ==="
stat -c "%y  %n" "$RD/checkpoints/hexgt_rl_latest.pt" 2>/dev/null
echo "=== supervisor.log control lines (tail) ==="
grep -E "LAUNCH|HALT|EXIT|COMPLETED|RELAUNCH|ABORT|SUPERVISOR" "$RD/supervisor.log" 2>/dev/null | tail -8
echo "=== running procs ==="
ps -eo pid,etime,cmd | grep -E "_rl_train\.py|_rl_supervise\.sh" | grep -v grep || echo "(none running)"
