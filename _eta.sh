#!/usr/bin/env bash
# Non-disturbing: report sweep process elapsed time + current GPU footprint.
p=$(pgrep -f _sweep_active.py | head -1)
if [ -n "$p" ]; then
  echo "sweep pid=$p elapsed=$(ps -o etime= -p "$p" | tr -d ' ')  etimes=$(ps -o etimes= -p "$p" | tr -d ' ')s"
else
  echo "sweep process NOT found (may have finished)"
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
