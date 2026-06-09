ROOT=/mnt/e/Hexo-BotTrainer-hexgt
cd /mnt/e/Hexo-BotTrainer || { echo "MIRROR CWD MISSING"; exit 1; }
PP="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
pkill -f 'hexo_frontend.web.*--port 8080' 2>/dev/null; sleep 1
nohup env CUDA_VISIBLE_DEVICES= HEXO_DEBUG_RUN_ROOT="$ROOT" PYTHONPATH="$PP" \
  /root/.venvs/hexgt-build/bin/python -m hexo_frontend.web --host 0.0.0.0 --port 8080 \
  > "$ROOT/runs/hexgnn_rl_main1/dashboard_8080.log" 2>&1 </dev/null &
disown
sleep 5
echo "=== curl / (code) ==="; curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/
echo "=== dashboard log tail ==="; tail -6 "$ROOT/runs/hexgnn_rl_main1/dashboard_8080.log"
echo "=== run-list API endpoints in web.py ==="; grep -nE "add_(get|route)|@.*get\(|'/api/" "$ROOT/packages/hexo_frontend/python/hexo_frontend/web.py" 2>/dev/null | grep -iE "run|training|diagnost|history" | head
