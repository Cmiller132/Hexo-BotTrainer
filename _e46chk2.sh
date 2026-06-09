RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "now: $(date '+%H:%M:%S')"
echo "epoch-46 shards: $(ls "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | wc -l)/512  newest: $(ls -t "$RD/selfplay"/epoch_000046_game_*.npz 2>/dev/null | head -1 | xargs -n1 stat -c '%y' 2>/dev/null | cut -c12-19)"
echo "rl_train.log tail (3):"; tail -3 "$RD/rl_train.log" | sed -E 's/\[[0-9: -]+\] //'
echo "driver etime: $(ps -eo etime,cmd | grep [_]rl_train_hexgnn | grep -v grep | grep -oE '^ *[0-9:]+' | head -1)"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
# live selfplay status if present
ls -t "$RD"/driver.*.out.log 2>/dev/null | head -1 | xargs tail -3 2>/dev/null | sed -E 's/\[[0-9: -]+\] //'
