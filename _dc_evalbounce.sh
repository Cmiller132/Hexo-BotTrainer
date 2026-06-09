RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "latest ckpt (resume target): $(ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
echo "bounce: eval_every=3 (owner) $(date -u +%FT%TZ)" > "$RD/supervisor_halted.flag"
SPID=$(ps -eo pid,cmd | grep "_dc_supervise_main1.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill rc=$?"; } || echo "(no supervisor running — was it down?)"
sleep 4
ps -eo pid,cmd | grep -E "_dc_supervise_main1|train_model.*dense_cnn_rl_main1" | grep -v grep || echo "dc supervisor+driver GONE"
for i in 1 2 3 4 5; do apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); echo "gpu $i:[$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3; done
echo "=== clear halt + relaunch (picks up edited evaluation.py + config) ==="
rm -f "$RD/supervisor_halted.flag" "$RD/supervisor.lock"
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
