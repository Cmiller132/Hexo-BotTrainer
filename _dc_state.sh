RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "supervisor/driver procs:"; ps -eo pid,pgid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep
echo "checkpoints: $(ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | wc -l) | selfplay shards: $(ls "$RD/selfplay"/ 2>/dev/null | wc -l)"
echo "halt flag: $(ls "$RD/supervisor_halted.flag" 2>/dev/null || echo none)"
