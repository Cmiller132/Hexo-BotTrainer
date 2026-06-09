#!/usr/bin/env bash
cd /mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_train/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python
python _dashboard_bridge.py --once 2>&1 | tail -2
echo "=== mirror selfplay .hxr ==="
ls -la --time-style=+%H:%M:%S /mnt/e/Hexo-BotTrainer/runs/hexgt_rl_main2/selfplay/*.hxr
