RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "epoch-30 shards: $(ls "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | wc -l)/512"
echo "newest shard mtime: $(ls -t "$RD/selfplay"/epoch_000030_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null)"
echo "driver alive: $(ps -eo pid,etime,cmd | grep _rl_train_hexgnn | grep -v grep | grep -oE '\-\-visits [0-9]+' | head -1)"
tail -2 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
