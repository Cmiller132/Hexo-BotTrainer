RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "now: $(date '+%H:%M:%S')  driver etime: $(ps -eo etime,cmd|grep [h]exo_train.cli.train_model|grep -v grep|grep -oE '^ *[0-9:]+'|head -1)"
echo "=== current phase signals ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('SELFPLAY live: epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'/',d.get('requested_games'),'active',d.get('active_games'),'pos/s',round(d.get('positions_per_second',0),1),'elapsed',round(d.get('elapsed_seconds',0)))" 2>/dev/null
echo "SealBot eval worker running? -> $(ps -eo etime,cmd|grep [_]sealbot_worker|grep -v grep|head -1|cut -c1-40 || echo NO)"
echo "=== recent train.out.log activity (which phase) ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null|head -1); tail -4 "$LOG" 2>/dev/null|grep -vE "frombuffer|writable|tensor_new"|cut -c1-100
echo "=== epoch-7 mtime timeline (resumed 02:43:10) ==="
echo "epoch6 result (epoch7 START ref): $(stat -c %y "$RD/diagnostics/epoch_000006.json" 2>/dev/null|cut -c12-19)"
echo "epoch7 selfplay-diag (selfplay END): $(stat -c %y "$RD/diagnostics/dense_cnn.selfplay.epoch_000007.json" 2>/dev/null|cut -c12-19 || echo '(not yet - selfplay still running)')"
echo "epoch7 shuffleddata dir: $(ls -dt "$RD/shuffleddata"/*epoch_000007* 2>/dev/null|head -1|xargs -n1 stat -c %y 2>/dev/null|cut -c12-19 || echo none)"
echo "epoch7 eval-diag (eval END): $(stat -c %y "$RD/diagnostics/dense_cnn.evaluation.epoch_000007.json" 2>/dev/null|cut -c12-19 || echo '(not yet)')"
echo "epoch7 result (epoch END): $(stat -c %y "$RD/diagnostics/epoch_000007.json" 2>/dev/null|cut -c12-19 || echo '(not yet)')"
echo "=== what stage does the DASHBOARD label show? (the owner's view) ==="
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=8).read());s=d.get('status',{});print('status.stage=',s.get('stage'),'| stage_status=',s.get('stage_status'),'| current_epoch=',s.get('current_epoch'),'| selfplay_live.status=',(s.get('selfplay_live') or {}).get('status'))" 2>/dev/null
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)"
