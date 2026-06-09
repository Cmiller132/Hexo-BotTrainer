RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
SPID=$(ps -eo pid,cmd | grep "_rl_supervise_hexgnn.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill rc=$?"; } || echo "(no supervisor found — maybe already gone)"
sleep 4
echo "=== zero survivors? ==="
ps -eo pid,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep || echo "hexgnn supervisor+driver GONE"
echo "=== GPU freed? ==="
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "gpu try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU now: /'
echo "=== halt flag stays (no relaunch) ==="
cat "$RD/supervisor_halted.flag" 2>/dev/null && echo "[halt flag present]" || echo "[MISSING — would relaunch!]"
echo "bridge (kept, harmless): $(ps -eo pid -p 397 2>/dev/null | tail -1)"
