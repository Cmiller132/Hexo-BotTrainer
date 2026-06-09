ps -eo pid,pgid,etime,cmd | grep -iE "_rl_supervise_hexgnn|_rl_train_hexgnn|_dashboard_bridge_hexgnn" | grep -v grep
echo "--- current epoch tail ---"
tail -1 /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/rl_train.log | sed -E 's/\[[0-9: -]+\] //'
