RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "halt" > "$RD/supervisor_halted.flag"
SPID=$(ps -eo pid,cmd | grep "_dc_supervise_main1.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill rc=$?"; } || echo "(no supervisor)"
sleep 4
ps -eo pid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep || echo "GONE"
for i in 1 2 3 4 5; do apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); echo "gpu $i:[$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3; done
echo "=== wipe run dir + relaunch from prefit at games_per_epoch=256 ==="
rm -rf "$RD"; mkdir -p "$RD/checkpoints"
echo "prefit intact: $(ls /mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1_prefit/bootstrap_sealbot_prefit.pt 2>/dev/null && echo yes)"
tr -d '\r' < /mnt/e/Hexo-BotTrainer-hexgt/scripts/_dc_launch_main1.sh | bash
