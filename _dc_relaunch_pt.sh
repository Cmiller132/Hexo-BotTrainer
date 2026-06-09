RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== wipe run dir (prefit is in dense_cnn_rl_main1_prefit, untouched) ==="
rm -rf "$RD"; mkdir -p "$RD/checkpoints"
echo "run dir cleared. prefit still present: $(ls -la /mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1_prefit/bootstrap_sealbot_prefit.pt 2>/dev/null | awk '{print $5" bytes"}')"
echo "=== relaunch RL from pretrained init ==="
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
