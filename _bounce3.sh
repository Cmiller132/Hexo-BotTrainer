RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
# Dynamically resolve the supervisor's PGID (don't hardcode a stale pid).
SPID=$(ps -eo pid,cmd | grep "_rl_supervise_hexgnn.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
if [ -z "$PGID" ]; then echo "ERROR: no supervisor found"; exit 1; fi
echo "=== kill -9 -$PGID (supervisor+driver); bridge kept ==="
kill -9 -"$PGID" 2>&1; echo "rc=$?"
sleep 4
echo "=== survivors? ==="
ps -eo pid,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep || echo "supervisor+driver GONE"
echo "bridge: $(ps -o pid,etime -p 397 2>/dev/null | tail -1)"
echo "=== GPU clearing ==="
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU CLEAR"; break; }; sleep 3
done
rm -f "$RD/supervisor.lock"
ls "$RD/supervisor_halted.flag" 2>/dev/null && echo "HALT FLAG PRESENT" || echo "no halt flag (good)"
echo "latest.pt resume target: $(ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
