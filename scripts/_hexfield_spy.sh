#!/bin/bash
# Dump stacks of the prefit main process and one worker (diagnostic).
MAIN=$(ps aux | grep '[_]hexfield_prefit' | grep -v 'bash' | awk '{print $2}' | head -1)
echo "=== main pid: ${MAIN:-NONE} ==="
if [ -n "$MAIN" ]; then
  /root/.venvs/hexfield-dev/bin/py-spy dump --pid "$MAIN" 2>&1 | head -35
  echo "=== children ==="
  ps --ppid "$MAIN" -o pid,etime,%cpu,cmd 2>/dev/null | head -8
fi
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== log bytes ==="
wc -c /mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_1.log
