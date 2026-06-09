#!/usr/bin/env bash
echo '=== rl/monitor procs ==='
pgrep -af '_rl_train.py|_rl_supervise|_rl_run_fg|_resmon|_dashboard_bridge' | grep -v pgrep || echo 'none'
echo '=== GPU ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo '=== free RAM ==='
awk '/MemAvailable/{printf "%.1f GB free / ", $2/1048576} /MemTotal/{printf "%.1f GB total\n", $2/1048576}' /proc/meminfo
