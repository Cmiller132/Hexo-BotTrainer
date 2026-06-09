RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== NOW: $(date '+%H:%M:%S') | driver etime: $(ps -eo etime,cmd | grep [h]exo_train.cli.train_model | grep -v grep | grep -oE '^ *[0-9:]+' | head -1) ==="
echo "=== PROCESS STATE: sealbot eval worker? selfplay? ==="
ps -eo pid,etime,cmd | grep -iE "sealbot|_sealbot_worker" | grep -v grep | head -2 | cut -c1-90 || echo "(no sealbot worker)"
echo "driver %cpu/state: $(ps -eo %cpu,stat,cmd | grep [h]exo_train.cli.train_model | grep -v grep | cut -c1-30)"
echo "=== LIVE self-play status (epoch, running?, elapsed, pos/s) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print({k:d.get(k) for k in ('status','epoch','games_started','games_finished','active_games','positions_per_second','elapsed_seconds')})" 2>/dev/null
echo "=== per-epoch phase summary (selfplay done? eval done? train skipped?) ==="
for ep in 000001 000002 000003 000004; do
  sp="$RD/diagnostics/dense_cnn.selfplay.epoch_$ep.json"; ev="$RD/diagnostics/dense_cnn.evaluation.epoch_$ep.json"; e="$RD/diagnostics/epoch_$ep.json"
  [ -f "$e" ] && $V -c "import json;d=json.load(open('$e'));r=d['metadata']['result'];print('epoch $ep: status=',d.get('status'),'train=',r.get('training',{}).get('status'),'eval=',r.get('evaluation',{}).get('status'),'wins=',r.get('evaluation',{}).get('wins'),'games=',r.get('evaluation',{}).get('games'))" 2>/dev/null || echo "epoch $ep: (no epoch_$ep.json yet)"
done
echo "=== checkpoints + shuffleddata (warmup progress toward 100k) ==="
ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
$V -c "import json,glob,os
tot=0
for f in sorted(glob.glob('$RD/diagnostics/dense_cnn.selfplay.epoch_*.json')):
    d=json.load(open(f)); tot+=int(d.get('raw_samples') or 0)
print('cumulative raw_samples across epochs:',tot,'/ 100000 shuffle_min_rows')" 2>/dev/null
echo "=== events.jsonl last 3 stages ==="
tail -3 "$RD/diagnostics/events.jsonl" 2>/dev/null | $V -c "import sys,json;[print(' ',json.loads(l).get('payload',{}).get('stage') or json.loads(l).get('event')) for l in sys.stdin if l.strip()]" 2>/dev/null
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null
