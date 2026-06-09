PP="/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python"
PID=$(ps -eo pid,cmd | grep "[h]exo_frontend.web" | grep -v grep | awk '{print $1}' | head -1)
echo "killing old dashboard pid=$PID"; kill "$PID" 2>/dev/null; sleep 2; kill -9 "$PID" 2>/dev/null; sleep 1
cd /mnt/e/Hexo-BotTrainer
setsid env CUDA_VISIBLE_DEVICES= HEXO_DEBUG_RUN_ROOT=/mnt/e/Hexo-BotTrainer-hexgt PYTHONPATH="$PP" \
  /root/.venvs/hexgt-build/bin/python -m hexo_frontend.web --host 0.0.0.0 --port 8080 \
  > /mnt/e/Hexo-BotTrainer/_dash_8080.log 2>&1 &
echo "relaunched; waiting to serve..."
for i in $(seq 1 15); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/ 2>/dev/null)
  [ "$code" = "200" ] && { echo "dashboard serving HTTP 200"; break; }
  sleep 2
done
ps -eo pid,etime,cmd | grep "[h]exo_frontend.web" | grep -oE "^ *[0-9]+ +[0-9:]+" | head -1
