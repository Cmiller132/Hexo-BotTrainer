echo "=== hexo_frontend.web processes ==="
ps -eo pid,pgid,etime,cmd | grep "[h]exo_frontend.web" | grep -v grep | cut -c1-60
echo "=== dashboard serving? ==="
echo "localhost:8080 -> HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)"
echo "=== dc run still visible (stopped run history accessible)? ==="
curl -s "http://127.0.0.1:8080/api/training/runs" 2>/dev/null | grep -o "dense_cnn_rl_main1" | head -1
echo "=== final dc process check (definitely gone) ==="
ps -eo pid,cmd | grep -E "_dc_supervise_main1|train_model.*dense_cnn_rl_main1" | grep -v grep || echo "dc run: GONE"
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed 's/^/gpu compute apps: /' ; echo "(empty above = GPU free)"
