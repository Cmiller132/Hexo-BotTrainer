RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== (1) live self-play: games finished? ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','requested_games','games_started','completed_games','games_finished','active_games','searched_positions','positions_per_second','elapsed_seconds')})" 2>/dev/null || echo "(no live json)"
echo "=== (2) on-disk records: selfplay dir contents ==="
ls -la "$RD/selfplay"/ 2>/dev/null | head
echo "  .hxr files: $(ls "$RD/selfplay"/*.hxr 2>/dev/null | wc -l) | per-game npz: $(ls "$RD/selfplay"/*game*.npz 2>/dev/null | wc -l) | other: $(ls "$RD/selfplay"/ 2>/dev/null | grep -vcE "hxr|game.*npz")"
echo "  newest selfplay file mtime: $(ls -t "$RD/selfplay"/ 2>/dev/null | head -1)"
echo "=== (3) checkpoints / epoch progress ==="
echo "checkpoints: $(ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | wc -l)"
echo "=== how HEXGNN selfplay dir looked (for comparison) ==="
ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/selfplay/ 2>/dev/null | sed -nE '1,3p;$p'
echo "  hexgnn .hxr: $(ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/selfplay/*.hxr 2>/dev/null | wc -l) | game npz: $(ls /mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1/selfplay/*game*.npz 2>/dev/null | wc -l)"
