RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "owner STOP directive 2026-06-06: dense_cnn_rl_main1 stopped at epoch-6 boundary; resumable; do not relaunch" > "$RD/supervisor_halted.flag"
SPID=$(ps -eo pid,cmd | grep "_dc_supervise_main1.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID (dashboard NOT touched)"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill -9 -$PGID rc=$?"; } || echo "(no supervisor found)"
sleep 4
echo "=== zero survivors (supervisor+driver)? ==="
ps -eo pid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep || echo "dense_cnn supervisor+driver GONE"
echo "=== GPU freed? (poll) ==="
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  echo "try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU now: /'
echo "=== resumability: latest checkpoint on disk ==="
ls -la "$RD/checkpoints"/epoch_000006.pt 2>/dev/null | awk '{print "  epoch_000006.pt:",$5,"bytes",$6,$7,$8}'
ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename | sed 's/^/  latest = /'
echo "=== halt flag stays ==="; cat "$RD/supervisor_halted.flag" 2>/dev/null
echo "=== dashboard still alive (not killed)? ==="; ps -eo pid,etime -p $(ps -eo pid,cmd | grep "[h]exo_frontend.web" | grep -v grep | awk '{print $1}' | head -1) 2>/dev/null | tail -1
