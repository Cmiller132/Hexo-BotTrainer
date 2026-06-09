RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 300
echo "=== RESUME from epoch 6? ==="
$V -c "import json;r=json.load(open('$RD/diagnostics/load_checkpoint.json'))['metadata']['result'];print('status=',r.get('status'),'checkpoint_ref=',r.get('checkpoint_ref'),'epoch=',r.get('epoch'))" 2>/dev/null
echo "=== config intact ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));a=d['model']['config']['architecture'];s=d['model']['config']['selfplay'];print('channels=',a['channels'],'blocks=',a['residual_blocks'],'visits=',s['search_visits'],'active=',s['active_games'])" 2>/dev/null
echo "=== self-play producing + games_per_epoch (requested) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','requested_games','games_started','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo "=== errors + GPU + driver ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1); grep -iE "Traceback|Error|Exception|shape|mismatch" "$LOG" 2>/dev/null | grep -viE "stderr=" | head -2 || echo "errors: (none)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
ps -eo cmd | grep -q "[h]exo_train.cli.train_model" && echo "driver ALIVE" || echo "driver GONE"
echo "=== DASHBOARD up? (restart if WSL restart killed it) ==="
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)
if [ "$code" != "200" ]; then
  echo "dashboard DOWN (HTTP $code) -> restarting"
  PP="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python"
  P=$(ps -eo pid,cmd|grep '[h]exo_frontend.web'|grep -v grep|awk '{print $1}'); [ -n "$P" ] && kill -9 $P 2>/dev/null
  cd /mnt/e/Hexo-BotTrainer
  setsid env CUDA_VISIBLE_DEVICES= HEXO_DEBUG_RUN_ROOT=/mnt/e/Hexo-BotTrainer-hexgt PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -m hexo_frontend.web --host 0.0.0.0 --port 8080 > /mnt/e/Hexo-BotTrainer/_dash_8080.log 2>&1 &
  for i in $(seq 1 12); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)" = "200" ] && break; sleep 2; done
fi
echo "dashboard: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)"
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read());s=d.get('status',{});print('dashboard shows: stage=',s.get('stage'),'selfplay_live=',(s.get('selfplay_live') or {}).get('status'))" 2>&1 | tail -1
