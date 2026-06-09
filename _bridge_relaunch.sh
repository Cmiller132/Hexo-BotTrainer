#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt; VENV=/root/.venvs/hexgt-build
pkill -TERM -f "_dashboard_bridge.py" 2>/dev/null; sleep 2
setsid bash -c "cd '$ROOT'; source '$VENV/bin/activate'; exec python '$ROOT/_dashboard_bridge.py'" >> "$ROOT/runs/hexgt_rl_main/_bridge.log" 2>&1 &
echo "bridge pid $!"
