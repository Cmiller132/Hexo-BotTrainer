#!/usr/bin/env bash
for p in $(pgrep -f _dashboard_bridge.py); do echo "killing $p"; kill -9 "$p"; done
sleep 2
pgrep -af _dashboard_bridge.py | grep -v pgrep || echo "all bridges stopped"
