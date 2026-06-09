RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "halt (pretrain relaunch)" > "$RD/supervisor_halted.flag"
SPID=$(ps -eo pid,cmd | grep "_dc_supervise_main1.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill rc=$?"; } || echo "(no supervisor)"
sleep 4
ps -eo pid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep || echo "dense_cnn supervisor+driver GONE"
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "gpu try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU now: /'
