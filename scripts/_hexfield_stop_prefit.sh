#!/bin/bash
# Stop the BC prefit (3 passes done) to free the GPU for M8+ (owner-authorized
# GPU management for the hexfield build).
MAIN=$(ps aux | grep '[_]hexfield_prefit.py' | awk '{print $2}')
echo "prefit pids: ${MAIN:-none}"
for p in $MAIN; do kill "$p" 2>/dev/null && echo "TERM $p"; done
# DataLoader workers re-parent to init; they exit when the main dies, but give
# them a moment then report GPU.
sleep 10
echo "--- gpu after stop:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
