RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
SPID=$(ps -eo pid,cmd | grep "_rl_supervise_hexgnn.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
[ -z "$PGID" ] && { echo "ERROR: no supervisor"; exit 1; }
kill -9 -"$PGID" 2>&1; echo "kill rc=$?"
sleep 4
ps -eo pid,cmd | grep -E "_rl_train_hexgnn|_rl_supervise_hexgnn" | grep -v grep || echo "supervisor+driver GONE"
echo "bridge: $(ps -o pid,etime -p 397 2>/dev/null | tail -1)"
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "gpu try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU CLEAR"; break; }; sleep 3
done
rm -f "$RD/supervisor.lock"
ls "$RD/supervisor_halted.flag" 2>/dev/null && echo "HALT FLAG PRESENT" || echo "no halt flag (good)"
echo "resume target: $(ls -t "$RD/checkpoints"/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
