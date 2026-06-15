#!/bin/bash
MAIN=$(ps aux | grep 'hexfield_smoke' | grep -v grep | grep -v 'bash -c' | awk '{print $2}' | head -1)
echo "pid: ${MAIN:-NONE}"
if [ -n "$MAIN" ]; then
  for i in 1 2 3; do
    /root/.venvs/hexfield-dev/bin/py-spy dump --pid "$MAIN" 2>&1 | grep -A20 "Thread.*MainThread" | head -22
    echo "---"
    sleep 3
  done
fi
