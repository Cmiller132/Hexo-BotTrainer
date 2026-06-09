#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt
VENV=/root/.venvs/hexgt-build
cd "$ROOT" || exit 11
# gentle SIGTERM to the OLD bridge (my read-only process), then relaunch updated one
kill -TERM 2150424 2>/dev/null; pkill -TERM -f "_dashboard_bridge.py" 2>/dev/null
sleep 2
source "$VENV/bin/activate"
echo "=== run updated bridge once (writes worktree + MAIN tree) ==="
python _dashboard_bridge.py --once 2>&1 | tail -2
echo "=== MAIN tree diagnostics now present? ==="
ls -la /mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main/diagnostics/ 2>/dev/null
echo "=== relaunch bridge loop (setsid, detached) ==="
setsid bash -c "cd '$ROOT'; source '$VENV/bin/activate'; exec python '$ROOT/_dashboard_bridge.py'" >> "$ROOT/runs/hexgt_rl_main/_bridge.log" 2>&1 &
echo "bridge relaunched pid $!"
