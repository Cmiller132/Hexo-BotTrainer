PP="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python"
PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -c "import hexo_frontend.web" 2>&1 | tail -2 && echo "import OK"
PID=$(ps -eo pid,cmd | grep "[h]exo_frontend.web" | grep -v grep | awk '{print $1}' | head -1)
kill "$PID" 2>/dev/null; sleep 2; kill -9 "$PID" 2>/dev/null; sleep 1
cd /mnt/e/Hexo-BotTrainer
setsid env CUDA_VISIBLE_DEVICES= HEXO_DEBUG_RUN_ROOT=/mnt/e/Hexo-BotTrainer-hexgt PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -m hexo_frontend.web --host 0.0.0.0 --port 8080 > /mnt/e/Hexo-BotTrainer/_dash_8080.log 2>&1 &
for i in $(seq 1 15); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)" = "200" ] && { echo "serving 200"; break; }; sleep 2; done
V=/root/.venvs/hexgt-build/bin/python; DR=/mnt/e/Hexo-BotTrainer/runs; P="$DR/_loss_probe"; mkdir -p "$P/diagnostics"
echo '{"status":"completed","elapsed_seconds":10,"metadata":{"result":{"epoch":1,"training":{"status":"completed","loss":5.12,"steps":512,"loss_components":{"policy":3.10,"value":1.00,"opp_policy":0.50,"stvalue_1":0.30,"stvalue_4":0.25,"stvalue_8":0.20}}}}}' > "$P/diagnostics/epoch_000001.json"
echo '{"status":"completed","epoch":1,"requested_games":256,"completed_games":256,"games_finished":256,"positions_per_second":12.0,"searched_positions":21000}' > "$P/diagnostics/dense_cnn.selfplay.epoch_000001.json"
echo "=== PROBE selfplay.buffer (should now have the per-head loss band) ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=_loss_probe',timeout=10).read());print(d['epoch_history'][-1]['selfplay'].get('buffer'))" 2>&1 | tail -2
echo "=== hexgnn still intact (no regression) ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=hexgnn_rl_main1',timeout=10).read());rows=[r for r in d['epoch_history'] if (r.get('selfplay') or {}).get('buffer')];print('buffers:',len(rows),'last:',{k:v for k,v in rows[-1]['selfplay']['buffer'].items() if 'loss' in k} if rows else None)" 2>&1 | tail -2
echo "=== dense_cnn skipped epochs still graceful (no buffer) ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read());print([(r['epoch'],r.get('training',{}).get('status'),(r.get('selfplay') or {}).get('buffer')) for r in d['epoch_history']])" 2>&1 | tail -2
rm -rf "$P"; echo "probe removed"
