echo "=== ensure no lingering dense_cnn/train_model proc + GPU clear ==="
ps -eo pid,cmd | grep -E "[h]exo_train.cli.train_model" | head || echo "none"
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed 's/^/gpu apps: /' || echo "gpu apps: none"
echo "=== LAUNCH dense_cnn_rl_main1 ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
