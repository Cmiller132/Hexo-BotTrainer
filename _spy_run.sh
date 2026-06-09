#!/usr/bin/env bash
pid=$(ps -eo pid,cmd | grep '_hexgnn_posps.py' | grep -v grep | awk '{print $1}' | head -1)
echo "pid=[$pid]"
if [ -z "$pid" ]; then
  echo "no posps python found; python procs:"; ps -eo pid,cmd | grep -i python | grep -v grep | head
  exit 0
fi
/root/.venvs/hexgt-build/bin/py-spy record --pid "$pid" --native --duration 40 --rate 200 --format raw -o /mnt/e/Hexo-BotTrainer-hexgt/_spy.folded 2>/mnt/e/Hexo-BotTrainer-hexgt/_spy.err
echo "py-spy exit=$?"
echo "folded lines:"; wc -l /mnt/e/Hexo-BotTrainer-hexgt/_spy.folded 2>/dev/null
echo "--- err tail ---"; tail -3 /mnt/e/Hexo-BotTrainer-hexgt/_spy.err
