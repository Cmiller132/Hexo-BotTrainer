#!/usr/bin/env bash
set -uo pipefail
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "=== full nvidia-smi top ==="
nvidia-smi
echo "=== any python procs at all ==="
pgrep -af python || echo "(no python)"
