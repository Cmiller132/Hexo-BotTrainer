RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "=== rl_train.log tail ==="; tail -4 "$RD/rl_train.log" 2>/dev/null
echo "=== epoch-0 self-play shards so far ==="; ls "$RD/selfplay"/epoch_000000_game_*.npz 2>/dev/null | wc -l
echo "=== baseline eval done? ==="; ls "$RD/eval/baseline_bc_eval.json" 2>/dev/null && echo "(baseline written)" || echo "(baseline not yet)"
echo "=== bridge.log tail ==="; tail -3 "$RD/bridge.log" 2>/dev/null
