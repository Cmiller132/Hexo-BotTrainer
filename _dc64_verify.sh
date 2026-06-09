RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 360
echo "=== errors? ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)
grep -iE "Traceback|Error|Exception|FAILED|shape|mismatch|OOM|CUDA error" "$LOG" 2>/dev/null | grep -viE "no error|stderr=" | head -5 || echo "(none)"
echo "=== 64x10 arch confirmation (config.normalized) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));a=d.get('model',{}).get('config',{}).get('architecture',d.get('architecture',{}));print('channels:',a.get('channels'),'residual_blocks:',a.get('residual_blocks'),'input_channels:',a.get('input_channels'))" 2>/dev/null || grep -hoE "\"channels\": *[0-9]+|\"residual_blocks\": *[0-9]+" "$RD/diagnostics/config.normalized.json" 2>/dev/null
echo "=== from-scratch (load_checkpoint) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/load_checkpoint.json'));print('result:',d.get('metadata',{}).get('result'))" 2>/dev/null
echo "=== live self-play (pos/s, games) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','active_games','searched_positions','positions_per_second','elapsed_seconds')})" 2>/dev/null || echo "(live json not yet written)"
echo "=== GPU + run dir ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
ls "$RD/selfplay"/ 2>/dev/null | head -3; echo "checkpoints: $(ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | wc -l)"
