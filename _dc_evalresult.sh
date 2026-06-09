RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch-0 SealBot eval (full) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.evaluation.epoch_000001.json'));print(json.dumps(d,indent=1)[:900])" 2>/dev/null
echo "=== current phase (epoch 1 self-play) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
