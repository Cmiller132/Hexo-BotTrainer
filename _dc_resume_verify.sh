RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 300
echo "=== RESUME confirmation: load_checkpoint (should be epoch_000006.pt, NOT prefit, NOT random) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/load_checkpoint.json'));r=d.get('metadata',{}).get('result',{});print('status=',r.get('status'),'checkpoint_ref=',r.get('checkpoint_ref'),'epoch=',r.get('epoch'))" 2>/dev/null
echo "=== config intact (64x10, visits, active) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));a=d['model']['config']['architecture'];s=d['model']['config']['selfplay'];print('channels=',a['channels'],'blocks=',a['residual_blocks'],'search_visits=',s['search_visits'],'active_games=',s['active_games'])" 2>/dev/null
echo "=== self-play producing games? (resumed epoch 7) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','requested_games','games_started','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo "=== errors? ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1); grep -iE "Traceback|Error|Exception|shape|mismatch|FAILED" "$LOG" 2>/dev/null | grep -viE "stderr=" | head -3 || echo "(none)"
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
echo "=== driver alive + games/epoch (live requested_games=256 confirms it) ==="
ps -eo cmd | grep -q "[h]exo_train.cli.train_model" && echo "driver ALIVE" || echo "driver GONE"
