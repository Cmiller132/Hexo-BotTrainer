RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== load_checkpoint (from-scratch confirmation) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/load_checkpoint.json'));print(json.dumps(d,indent=0)[:600])" 2>/dev/null
echo "=== initialize_run (params/arch/visits/seed) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/initialize_run.json'));import sys;[print(k,'=',v) for k,v in d.items() if k in('parameters','params','total_parameters','channels','residual_blocks','search_visits','seed','device','model','architecture')] or print(json.dumps(d)[:700])" 2>/dev/null
echo "=== calibration (pos/s + chosen batches) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.performance_calibration.json'));print(json.dumps(d)[:600])" 2>/dev/null
echo "=== live self-play (epoch 0 pos/s) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print(json.dumps(d)[:600])" 2>/dev/null
