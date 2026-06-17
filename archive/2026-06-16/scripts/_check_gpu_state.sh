#!/usr/bin/env bash
# Diagnostic: report running dense_cnn procs + GPU state.
set -uo pipefail

echo "=== dense_cnn / supervisor / trainer processes ==="
pgrep -af "dense_cnn|supervise_target|train_model|bootstrap_dense_cnn" || echo "(none)"

echo
echo "=== nvidia-smi (memory + util) ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi failed"

echo
echo "=== compute apps on GPU ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true
