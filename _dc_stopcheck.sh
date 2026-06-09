RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "now: $(date '+%H:%M:%S')"
echo "=== latest checkpoint on disk (resumability) ==="
ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename 2>/dev/null
ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
echo "=== current phase: self-play / eval / train? ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('live selfplay epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'/',d.get('requested_games'),'pos/s',round(d.get('positions_per_second',0),1),'elapsed',round(d.get('elapsed_seconds',0)))" 2>/dev/null
echo "sealbot eval worker running? $(ps -eo etime,cmd | grep [_]sealbot_worker | grep -v grep | head -1 | cut -c1-40 || echo no)"
echo "driver etime: $(ps -eo etime,cmd | grep [h]exo_train.cli.train_model | grep -v grep | grep -oE '^ *[0-9:]+' | head -1)"
echo "latest epoch result file: $(ls -t "$RD/diagnostics"/epoch_*.json 2>/dev/null | head -1 | xargs -n1 basename)"
echo "supervisor pgid:"; ps -eo pid,pgid,cmd | grep -E "_dc_supervise_main1|hexo_train.cli.train_model" | grep -v grep | head
echo "halt flag: $(ls "$RD/supervisor_halted.flag" 2>/dev/null || echo none)"
