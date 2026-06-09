#!/bin/bash
sleep 8
echo "=== 8080 dashboard proc + port ==="
ps -eo pid,etimes,args | grep "hexo_frontend.web" | grep -v grep | cut -c1-90
ss -ltnp 2>/dev/null | grep ":8080" || echo "8080 NOT listening yet"
echo "=== 8080 dashboard log tail ==="
tail -4 /mnt/e/Hexo-BotTrainer/runs/_dash8080.log 2>/dev/null
echo "=== sweep still running (untouched)? ==="
ps -o pid= -p 392 >/dev/null 2>&1 && echo "sweep pid 392 ALIVE" || echo "sweep 392 not found (check name)"
ps -eo pid,args | grep "_sweep_active" | grep -v grep | cut -c1-70 | head -1
echo "=== main-tree hexgt mirror present (for dashboard to show)? ==="
ls -d /mnt/e/Hexo-BotTrainer/runs/hexgt_rl* 2>/dev/null
