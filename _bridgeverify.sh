#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
echo "=== supervisor log: bridge watchdog line ==="
grep -E "SUPERVISOR start|bridge watchdog|LAUNCH" "$WT/runs/hexgt_rl_main2/supervisor.log" | tail -3
echo "=== process tree: supervisor -> driver + bridge watchdog -> bridge ==="
sup=$(pgrep -f "_rl_supervise" | head -1)
echo "supervisor pid=$sup"
echo "driver:        $(pgrep -af _rl_train.py | grep -v pgrep | head -1)"
echo "bridge:        $(pgrep -af _dashboard_bridge.py | grep -v pgrep | head -1)"
echo "bridge count (must be 1): $(pgrep -fc _dashboard_bridge.py)"
echo "=== bridge parent (should chain to supervisor $sup) ==="
bpid=$(pgrep -f _dashboard_bridge.py | head -1)
[ -n "$bpid" ] && ps -o pid,ppid,cmd -p "$bpid" && echo "ppid chain:" && pstree -p -s "$bpid" 2>/dev/null | head -1
echo "=== bridge.log tail (looping + writing?) ==="
tail -3 "$WT/runs/hexgt_rl_main2/bridge.log" 2>/dev/null
echo "=== mirror freshness (MAIN diagnostics newest file) ==="
ls -lat --time-style=+%H:%M:%S /mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main2/diagnostics/*.json 2>/dev/null | head -2
echo "=== driver RESUME line ==="
grep -E "RESUME|GRAFT" "$WT/runs/hexgt_rl_main2/rl_train.log" | tail -1
echo "=== resources ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
