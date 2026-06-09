DR=/mnt/e/Hexo-BotTrainer/runs; V=/root/.venvs/hexgt-build/bin/python
P="$DR/_loss_probe"; mkdir -p "$P/diagnostics"
# synthetic dense_cnn-format TRAINED epoch (training.loss + loss_components) + minimal selfplay diag
cat > "$P/diagnostics/epoch_000001.json" <<'J'
{"status":"completed","elapsed_seconds":10,"metadata":{"result":{"epoch":1,"training":{"status":"completed","loss":5.12,"steps":512,"loss_components":{"policy":3.10,"value":1.00,"opp_policy":0.50,"stvalue_1":0.30,"stvalue_4":0.25,"stvalue_8":0.20}}}}}
J
cat > "$P/diagnostics/dense_cnn.selfplay.epoch_000001.json" <<'J'
{"status":"completed","epoch":1,"requested_games":256,"completed_games":256,"games_finished":256,"positions_per_second":12.0,"searched_positions":21000}
J
echo "=== PROBE (synthetic trained dense_cnn epoch): does selfplay.buffer carry the loss band? ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=_loss_probe',timeout=10).read());sp=d['epoch_history'][-1].get('selfplay',{});print('selfplay.buffer =', sp.get('buffer'))" 2>&1 | tail -2
echo ""
echo "=== LIVE dense_cnn_rl_main1: skipped epochs graceful (training.status + no buffer)? ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read())
for r in d['epoch_history']:
    t=r.get('training',{}); sp=r.get('selfplay',{})
    print(' epoch',r['epoch'],'training.status=',t.get('status'),'loss=',t.get('loss'),'selfplay.buffer=',sp.get('buffer'))" 2>&1 | tail -4
echo ""
echo "=== hexgnn no-regression: loss band buffer still present? ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=hexgnn_rl_main1',timeout=10).read())
rows=[r for r in d['epoch_history'] if (r.get('selfplay') or {}).get('buffer')]
print(' epochs with selfplay.buffer:',len(rows),'| last buffer loss fields:', {k:v for k,v in (rows[-1]['selfplay']['buffer'].items() if rows else []) if 'loss' in k})" 2>&1 | tail -2
echo "=== cleanup probe ==="; rm -rf "$P"; echo "removed $P"
