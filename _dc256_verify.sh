RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 330
echo "=== games_per_epoch (config.normalized [selfplay]) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));sp=d.get('model',{}).get('config',{}).get('selfplay',{});print('selfplay.games_per_epoch=',sp.get('games_per_epoch'),'active_games=',sp.get('active_games'),'search_visits=',sp.get('search_visits'))" 2>/dev/null
echo "=== init-from-prefit ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/load_checkpoint.json'));r=d.get('metadata',{}).get('result',{});print('status=',r.get('status'),'checkpoint_ref=',r.get('checkpoint_ref'))" 2>/dev/null
echo "=== self-play: requested_games (should be 256) + clean active==games wave ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','requested_games','games_started','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo "=== arch + errors + GPU ==="
$V -c "import json;a=json.load(open('$RD/diagnostics/config.normalized.json'))['model']['config']['architecture'];print('channels',a['channels'],'blocks',a['residual_blocks'])" 2>/dev/null
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1); grep -iE "Traceback|Error|Exception|FAILED|hang" "$LOG" 2>/dev/null | grep -viE "stderr=" | head -3 || echo "errors: (none)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
