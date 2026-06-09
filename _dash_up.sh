echo "=== any hexo_frontend.web running? ==="
ps -eo pid,etime,cmd | grep "[h]exo_frontend.web" | grep -v grep | cut -c1-50 || echo "(none - dashboard down)"
PP="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python"
echo "=== preflight import ==="
PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -c "import hexo_frontend.web; print('import OK')" 2>&1 | tail -2
echo "=== (re)start dashboard ==="
# kill any stale instance first
P=$(ps -eo pid,cmd | grep "[h]exo_frontend.web" | grep -v grep | awk '{print $1}'); [ -n "$P" ] && kill -9 $P 2>/dev/null
cd /mnt/e/Hexo-BotTrainer
setsid env CUDA_VISIBLE_DEVICES= HEXO_DEBUG_RUN_ROOT=/mnt/e/Hexo-BotTrainer-hexgt PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -m hexo_frontend.web --host 0.0.0.0 --port 8080 > /mnt/e/Hexo-BotTrainer/_dash_8080.log 2>&1 &
for i in $(seq 1 15); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)" = "200" ] && { echo "dashboard serving HTTP 200 (pid $(ps -eo pid,cmd|grep '[h]exo_frontend.web'|grep -v grep|awk '{print $1}'))"; break; }; sleep 2; done
/root/.venvs/hexgt-build/bin/python -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read());s=d.get('status',{});print('dense_cnn_rl_main1 live: stage=',s.get('stage'),'selfplay_live=',(s.get('selfplay_live') or {}).get('status'),'epoch=',s.get('current_epoch'))" 2>&1 | tail -2
