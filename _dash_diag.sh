#!/bin/bash
echo "=== dashboard / bridge procs alive? ==="
ps -eo pid,etimes,args | grep -E "hexo_frontend.web|_dashboard_bridge" | grep -v grep | cut -c1-100
echo "(blank = none)"
echo "=== specific pids (728960=8080, 2150425=8081)? ==="
for p in 728960 2150425; do ps -o pid= -p $p >/dev/null 2>&1 && echo "pid $p ALIVE" || echo "pid $p DEAD"; done
echo "=== ports listening (8080/8081)? ==="
ss -ltnp 2>/dev/null | grep -E ":8080|:8081" || echo "neither 8080 nor 8081 listening"
echo "=== what IS using the GPU (the sweep — do NOT touch) ==="
ps -eo pid,etimes,args | grep -iE "python" | grep -v grep | grep -ivE "_dashboard_bridge|hexo_frontend" | cut -c1-90 | head
