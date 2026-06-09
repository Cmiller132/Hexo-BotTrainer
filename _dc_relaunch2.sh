RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
rm -f "$RD/supervisor_halted.flag" "$RD/supervisor.lock"
echo "halt flag cleared: $(ls "$RD/supervisor_halted.flag" 2>/dev/null || echo yes-cleared)"
echo "latest ckpt to resume from: $(ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
echo "=== relaunch ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
