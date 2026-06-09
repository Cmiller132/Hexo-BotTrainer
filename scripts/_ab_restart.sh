#!/usr/bin/env bash
# Clean restart of main3 from the seed with the optimized binary (TSS no-scan
# short-circuit always-on) + vbatch=128. Sets aside the partial epoch-0 shards
# produced by the OLD (slow, pre-opt) binary so epoch 0 regenerates fresh, clears
# the HALT flag, and relaunches the supervisor (setsid-detached, durable).
set -uo pipefail
RUN=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main3

echo "=== safety: nothing of ours still running? ==="
ps -eo pid,cmd | grep -E "[_]rl_train.py|[_]rl_supervise|[_]dashboard_bridge" | grep -v grep || echo "(clean)"

echo "=== set aside OLD partial epoch-0 artifacts (pre-opt binary) ==="
STAMP=$(cat "$RUN/.preopt_stamp" 2>/dev/null || echo preopt)
BK="$RUN/_preopt_$STAMP"
mkdir -p "$BK/selfplay" "$BK/eval"
shopt -s nullglob
mv "$RUN"/selfplay/epoch_000000_* "$BK/selfplay/" 2>/dev/null || true
mv "$RUN"/eval/epoch_000000_* "$BK/eval/" 2>/dev/null || true
echo "moved $(ls "$BK/selfplay" 2>/dev/null | wc -l) selfplay + $(ls "$BK/eval" 2>/dev/null | wc -l) eval files to $BK"

echo "=== clear HALT flag ==="
rm -f "$RUN/supervisor_halted.flag" "$RUN/supervisor.lock" "$RUN/supervisor_completed.flag"
ls "$RUN"/supervisor_halted.flag 2>/dev/null && echo "WARN halt still present" || echo "(halt cleared)"

echo "=== relaunch via _rl_launch_main3.sh (setsid-detached) ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_rl_launch_main3.sh | bash
echo "ALL_DONE"
