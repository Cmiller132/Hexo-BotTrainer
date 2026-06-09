#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/mnt/e/Hexo-BotTrainer-hexgt; VENV=/root/.venvs/hexgt-build
cd "$ROOT" || exit 11
pkill -TERM -f "_dashboard_bridge.py" 2>/dev/null; sleep 2
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python:$ROOT/packages/hexo_models/python:$ROOT/packages/hexo_frontend/python"
echo "=== run new bridge once ==="
python _dashboard_bridge.py --once 2>&1 | tail -5
M=/mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main
echo "=== MAIN/evaluation child dirs + hxr counts ==="
for d in "$M"/evaluation/*/; do echo "$(basename "$d"): $(ls "$d"*.hxr 2>/dev/null | wc -l) hxr"; done 2>/dev/null | head
echo "=== MAIN/selfplay hxr ==="; ls -la "$M"/selfplay/*.hxr 2>/dev/null
echo "=== MAIN/diagnostics epoch_*.json (loss detail) ==="; ls "$M"/diagnostics/epoch_0*.json 2>/dev/null
echo "=== relaunch loop ==="
setsid bash -c "cd '$ROOT'; source '$VENV/bin/activate'; export PYTHONPATH='$PYTHONPATH'; exec python '$ROOT/_dashboard_bridge.py'" >> "$ROOT/runs/hexgt_rl_main/_bridge.log" 2>&1 &
echo "bridge pid $!"
