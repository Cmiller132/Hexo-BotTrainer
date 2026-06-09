sup=$(pgrep -f '_rl_supervise.sh' | head -1)
pg=$(ps -o pgid= -p "$sup" | tr -d ' ')
echo "supervisor pid=$sup pgid=$pg"
kill -TERM -"$pg" 2>/dev/null
sleep 4
pgrep -af 'python.*_rl_train' | grep -v pgrep || echo NO_DRIVER
pgrep -af '_rl_supervise.sh' | grep -v pgrep || echo NO_SUP
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
