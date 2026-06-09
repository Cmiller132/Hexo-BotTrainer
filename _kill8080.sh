#!/usr/bin/env bash
for p in $(pgrep -f 'hexo_frontend.web.*--port 8080'); do echo "killing 8080 pid $p"; kill -9 "$p"; done
sleep 2
pgrep -f 'hexo_frontend.web.*--port 8080' >/dev/null && echo "STILL UP" || echo "8080 stopped"
