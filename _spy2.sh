pid=$(ps -eo pid,cmd | grep '_hexgnn_posps.py' | grep -v grep | awk '{print $1}' | head -1)
echo "pid=$pid"
timeout 30 /root/.venvs/hexgt-build/bin/py-spy dump --pid "$pid" 2>&1 | head -40
