echo "=== hexgnn processes (pid pgid sid cmd) ==="
ps -eo pid,pgid,sid,etime,cmd | grep -iE "_rl_supervise_hexgnn|_rl_train_hexgnn|_dashboard_bridge_hexgnn" | grep -v grep
echo ""
echo "=== driver.pid file ==="
cat runs/hexgnn_rl_main1/driver.pid 2>/dev/null
echo "=== flags present? ==="
ls -la runs/hexgnn_rl_main1/supervisor_halted.flag runs/hexgnn_rl_main1/supervisor.lock 2>/dev/null
echo "=== nvidia-smi compute apps ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
