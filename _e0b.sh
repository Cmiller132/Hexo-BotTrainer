RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "=== rl_train.log tail (30) ==="
tail -30 "$RD/rl_train.log" 2>/dev/null
echo "=== any pos/s lines so far ==="
grep -iE "pos/s|pos_s|positions/sec|selfplay.*epoch|epoch.*selfplay" "$RD/rl_train.log" 2>/dev/null | tail -8
echo "=== shard count epoch 0 ==="
ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l
