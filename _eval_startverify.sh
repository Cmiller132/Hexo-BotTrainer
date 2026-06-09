RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
sleep 180
echo "=== eval_every in normalized runtime config? ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/config.normalized.json'));e=d['model']['config']['evaluation'];print('eval_every=',e.get('eval_every'),'games_per_epoch=',e.get('games_per_epoch'),'sealbot_variant=',e.get('sealbot_variant'))" 2>/dev/null
echo "=== resume from epoch 25? ==="
$V -c "import json;r=json.load(open('$RD/diagnostics/load_checkpoint.json'))['metadata']['result'];print('status=',r.get('status'),'checkpoint_ref=',(r.get('checkpoint_ref') or '').split('/')[-1],'epoch=',r.get('epoch'))" 2>/dev/null
echo "=== self-play running (epoch 26)? + config 64x10/512/256 ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('selfplay epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'),'/',d.get('requested_games'),'pos/s',round(d.get('positions_per_second',0),1))" 2>/dev/null
$V -c "import json;a=json.load(open('$RD/diagnostics/config.normalized.json'))['model']['config'];print('channels',a['architecture']['channels'],'blocks',a['architecture']['residual_blocks'],'visits',a['selfplay']['search_visits'],'active',a['selfplay']['active_games'])" 2>/dev/null
echo "=== errors? GPU? ==="
LOG=$(ls -t "$RD"/train.*.out.log 2>/dev/null|head -1); grep -iE "Traceback|Error|Exception|eval_every|unknown key|rejected" "$LOG" 2>/dev/null | grep -viE "stderr=" | head -3 || echo "(no errors)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
