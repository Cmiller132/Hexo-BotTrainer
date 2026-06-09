RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 330
echo "=== INITIALIZED FROM PREFIT? (load_checkpoint checkpoint_ref should be the prefit, NOT null) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/load_checkpoint.json'));print('result:',d.get('metadata',{}).get('result'))" 2>/dev/null || echo "(no load_checkpoint.json yet)"
echo "=== 64x10 arch ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));a=d.get('model',{}).get('config',{}).get('architecture',{});print('channels:',a.get('channels'),'residual_blocks:',a.get('residual_blocks'))" 2>/dev/null
echo "=== self-play running? ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','active_games','searched_positions','positions_per_second','elapsed_seconds')})" 2>/dev/null || echo "(no live json yet)"
echo "=== errors? ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
grep -iE "Traceback|Error|Exception|shape|mismatch|FAILED" "$LOG" 2>/dev/null | grep -viE "stderr=" | head -4 || echo "(none)"
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
