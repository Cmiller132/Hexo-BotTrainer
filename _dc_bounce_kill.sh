RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== current state ==="
echo "latest checkpoint: $(ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
echo "all ckpts: $(ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
/root/.venvs/hexgt-build/bin/python -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('current phase: selfplay epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'/',d.get('requested_games'))" 2>/dev/null
echo "=== set halt flag + kill supervisor group ==="
echo "bounce relaunch (owner) $(date -u +%FT%TZ)" > "$RD/supervisor_halted.flag"
SPID=$(ps -eo pid,cmd | grep "_dc_supervise_main1.sh" | grep -v grep | awk '{print $1}' | head -1)
PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
echo "supervisor pid=$SPID pgid=$PGID (dashboard NOT touched)"
[ -n "$PGID" ] && { kill -9 -"$PGID" 2>&1; echo "kill rc=$?"; } || echo "(no supervisor)"
sleep 4
echo "=== zero survivors? ==="
ps -eo pid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep || echo "supervisor+driver GONE"
echo "=== GPU free? ==="
for i in 1 2 3 4 5; do apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); echo "try $i: [$apps]"; [ -z "$apps" ] && { echo "GPU FREE"; break; }; sleep 3; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
