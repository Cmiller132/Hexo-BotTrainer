RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch result files present ==="
ls "$RD/diagnostics"/epoch_*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
echo "=== selfplay diag files ==="; ls "$RD/diagnostics"/dense_cnn.selfplay.epoch_*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
echo "=== live + driver ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('live selfplay epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'pos/s',round(d.get('positions_per_second',0),1),'elapsed',round(d.get('elapsed_seconds',0)))" 2>/dev/null
echo "driver etime: $(ps -eo etime,cmd | grep [h]exo_train.cli.train_model | grep -v grep | grep -oE '^ *[0-9:]+' | head -1)"
echo "=== train status per epoch (which trained, loss) ==="
for f in $(ls "$RD/diagnostics"/epoch_*.json 2>/dev/null); do
  $V -c "import json;d=json.load(open('$f'));r=d['metadata']['result'];t=r.get('training',{});print('$(basename $f)','total_elapsed=',round(d.get('elapsed_seconds',0)),'train=',t.get('status'),'loss=',round(t.get('loss'),2) if t.get('loss') else None)" 2>/dev/null
done | tail -12
