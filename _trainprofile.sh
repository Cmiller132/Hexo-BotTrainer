RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch-5 training profile (steps/elapsed/throughput + per-head loss) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/epoch_000005.json'));t=d['metadata']['result'].get('training',{})
print('status=',t.get('status'),'loss=',round(t.get('loss',0),3),'steps=',t.get('steps'),'samples=',t.get('samples'),'batch=',t.get('batch_size'),'elapsed_s=',round(t.get('elapsed_seconds',0),1),'samples/s=',round(t.get('samples_per_second',0),1))
print('per-head loss_components=',{k:round(v,3) for k,v in (t.get('loss_components') or {}).items()})" 2>/dev/null
echo "=== LOSS PANEL: does the dashboard API now show epoch-5's real per-head loss (my fix)? ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read())
for r in d['epoch_history']:
    sp=r.get('selfplay') or {}; t=r.get('training') or {}
    if r['epoch']>=4: print(' epoch',r['epoch'],'train.status=',t.get('status'),'selfplay.buffer=',sp.get('buffer'))" 2>&1 | tail -4
echo "=== current phase after train (eval? next selfplay?) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('live selfplay epoch',d.get('epoch'),d.get('status'))" 2>/dev/null
ps -eo etime,cmd | grep -iE "_sealbot_worker" | grep -v grep | head -1 | cut -c1-50 && echo "(sealbot eval running)" || echo "(no eval worker)"
