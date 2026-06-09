RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "now: $(date '+%H:%M:%S')  driver etime: $(ps -eo etime,cmd | grep [_]rl_train_hexgnn | grep -v grep | grep -oE '^ *[0-9:]+' | head -1)"
echo "epoch-46 shards: $(ls "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | wc -l)/512  newest: $(ls -t "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null | cut -c12-19)"
echo "=== any epoch 46 line at all? (broad grep) ==="
grep -E "epoch 46" "$RD/rl_train.log" 2>/dev/null | tail -3 | sed -E 's/\[[0-9: -]+\] //'
echo "=== rl_train.log last 2 lines ==="
tail -2 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)"
