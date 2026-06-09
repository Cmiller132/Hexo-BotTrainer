#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
PY=/root/.venvs/hexgt-build/bin/python
# launch the hung min test in background
"$PY" scripts/_mcts_min2.py > /tmp/min2.out 2>&1 &
PID=$!
echo "launched pid=$PID"
sleep 16   # let it reach the hang (import+model+root eval)
echo "=== py-spy native dump (pid $PID) ==="
/root/.venvs/hexgt-build/bin/py-spy dump --pid $PID --native 2>&1 | head -60
kill -9 $PID 2>/dev/null
echo "=== min2 stdout tail ==="
grep -vE 'UserWarning|frombuffer|torch.csrc|warnings.warn' /tmp/min2.out | tail -8
