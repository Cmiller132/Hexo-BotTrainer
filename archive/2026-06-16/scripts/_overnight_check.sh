#!/usr/bin/env bash
# One-shot health snapshot for the dense_cnn_restnet_main1 relaunch.
set -uo pipefail
RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main1
echo "now_utc=$(date -u +%H:%M:%S)"
pid=$(cat "$RD/driver.pid" 2>/dev/null)
echo "driver.pid=$pid"
ps -p "$pid" -o pid,etime,rss,cmd --no-headers 2>/dev/null || echo "DRIVER NOT RUNNING"
echo "--- halt flag ---"
[ -f "$RD/supervisor_halted.flag" ] && echo "HALT FLAG PRESENT" || echo "no halt flag"
echo "--- newest diagnostics (last 6) ---"
ls -t "$RD/diagnostics" 2>/dev/null | head -6
echo "--- train log tail ---"
tail -3 "$RD"/train.20260610_032233.out.log 2>/dev/null
