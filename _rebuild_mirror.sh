#!/usr/bin/env bash
set -uo pipefail
# Stop the bridge, wipe ONLY the bridge-derived mirror artifacts (stale 0-indexed),
# then rebuild with the new 1-indexed naming. NEVER touches the run's real data
# (the worktree's selfplay/*.npz and eval/ are left untouched).
for p in $(pgrep -f _dashboard_bridge.py); do kill -9 "$p"; done
sleep 1

MAIN=/mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main2                 # main-tree mirror (derived)
WTDIAG=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/diagnostics  # bridge-written copy
echo "wiping derived mirror dirs (NOT the run's npz/eval)..."
rm -rf "$MAIN/selfplay" "$MAIN/evaluation" "$MAIN/diagnostics"
rm -rf "$WTDIAG"
echo "remaining in MAIN: $(ls "$MAIN" 2>/dev/null | tr '\n' ' ')"

cd /mnt/e/Hexo-BotTrainer-hexgt && source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python
python _dashboard_bridge.py --once 2>&1 | tail -3
echo "=== rebuilt mirror ==="
echo "selfplay hxr:"; ls "$MAIN/selfplay/" 2>/dev/null
echo "evaluation dirs:"; ls "$MAIN/evaluation/" 2>/dev/null
echo "diagnostics:"; ls "$MAIN/diagnostics/" 2>/dev/null
