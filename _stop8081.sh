#!/usr/bin/env bash
# Kill the worktree :8081 dashboard + the bridge (to reload MAIN-only DIAG_DIRS).
for p in $(pgrep -f 'hexo_frontend.web.*--port 8081'); do echo "killing 8081 pid $p"; kill -9 "$p"; done
for p in $(pgrep -f '_dashboard_bridge.py'); do echo "killing bridge pid $p"; kill -9 "$p"; done
sleep 2
echo "--- remaining hexo_frontend procs ---"
pgrep -af 'hexo_frontend.web' | grep -v pgrep
echo "--- 8081 down? ---"
curl -s -o /dev/null -w '8081 http=%{http_code}\n' localhost:8081 2>/dev/null || echo '8081 DOWN (good)'
