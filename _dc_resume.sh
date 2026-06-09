RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== pre-checks ==="
echo "halt flag: $(ls "$RD/supervisor_halted.flag" 2>/dev/null || echo none)"
echo "latest ckpt: $(ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
echo "GPU free: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) compute apps"
echo "no stale dc supervisor/driver: $(ps -eo pid,cmd | grep -E '_dc_supervise_main1|train_model.*dense_cnn_rl_main1' | grep -v grep | wc -l)"
echo "=== clear halt flag + remove stale lock ==="
rm -f "$RD/supervisor_halted.flag" "$RD/supervisor.lock"
echo "halt flag now: $(ls "$RD/supervisor_halted.flag" 2>/dev/null || echo cleared)"
echo "=== relaunch ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
