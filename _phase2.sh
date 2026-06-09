RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch 5 (000005): did it TRAIN? (first real loss?) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/epoch_000005.json'));r=d['metadata']['result']
t=r.get('training',{}); print('train.status=',t.get('status'),'reason=',t.get('reason'),'loss=',t.get('loss'),'steps=',t.get('steps'),'components=',list((t.get('loss_components') or {}).keys()))
s=r.get('samples',{}).get('selection',{}).get('metadata',{}).get('shuffle',{}); print('shuffle:',{k:s.get(k) for k in ('status','reason','output_rows','total_num_data_rows','used_rows')})
ev=r.get('evaluation',{}); print('eval.status=',ev.get('status'),'wins=',ev.get('wins'))" 2>/dev/null || echo "(no epoch_000005.json training yet — still in progress)"
echo "=== shuffleddata now (did the first shuffle produce data?) ==="
echo "shuffleddata npz: $(find "$RD/shuffleddata" -name '*.npz' 2>/dev/null | wc -l) | dirs: $(ls "$RD/shuffleddata" 2>/dev/null | head -3 | tr '\n' ' ')"
du -sh "$RD/shuffleddata" 2>/dev/null
echo "=== what's running NOW + how long (current-phase timing) ==="
ps -eo etime,cmd | grep -iE "sealbot|_sealbot_worker|shuffle" | grep -v grep | head -3 | cut -c1-80
echo "newest diagnostic written (what phase just produced output):"
ls -lt --time-style=+%H:%M:%S "$RD"/diagnostics/*.json 2>/dev/null | head -4 | awk '{print "  "$6" "$7}'
echo "now: $(date '+%H:%M:%S')"
echo "=== latest train.out.log tail (eval/train/shuffle activity) ==="
tail -6 "$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)" 2>/dev/null | grep -vE "frombuffer|writable|tensor_new" | cut -c1-110
