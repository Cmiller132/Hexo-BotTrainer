#!/bin/bash
# Confirm the eval arena is progressing (not hung): two stack samples + gpu.
MAIN=$(ps aux | grep 'hexfield_soak.toml' | grep -v grep | grep -v 'bash -c' | awk '{print $2}' | head -1)
echo "pid: ${MAIN:-NONE}  gpu:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
for i in 1 2; do
  /root/.venvs/hexfield-dev/bin/py-spy dump --pid "$MAIN" 2>/dev/null | grep -A7 MainThread | head -9
  echo "---"
  sleep 4
done
