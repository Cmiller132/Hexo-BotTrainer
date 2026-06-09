RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== per-phase elapsed (epochs 5 & 6): selfplay / train (direct) + total ==="
for ep in 000005 000006; do
  $V -c "import json;d=json.load(open('$RD/diagnostics/epoch_$ep.json'));r=d['metadata']['result']
sp=r.get('selfplay',{}).get('elapsed_seconds'); tr=r.get('training',{}).get('elapsed_seconds')
tot=d.get('elapsed_seconds'); resid=(tot or 0)-(sp or 0)-(tr or 0)
print('epoch $ep: TOTAL=%ss | selfplay=%ss | train=%ss | residual(shuffle+eval+ckpt)=%ss'%(round(tot or 0),round(sp or 0),round(tr or 0),round(resid)))
sh=r.get('samples',{}).get('selection',{}).get('metadata',{}).get('shuffle',{})
print('   shuffle: status=%s output_rows=%s used_rows=%s total_num_data_rows=%s desired_rows=%s'%(sh.get('status'),sh.get('output_rows'),sh.get('used_rows'),sh.get('total_num_data_rows'),sh.get('desired_rows')))
sd=r.get('selfplay',{}); print('   selfplay: games=%s pos/s=%s len? raw_samples=%s'%(sd.get('completed_games'),round(sd.get('positions_per_second') or 0,1),sd.get('raw_samples')))" 2>/dev/null
done
echo ""
echo "=== mtime timeline to separate SHUFFLE vs EVAL (residual split) for epoch 6 ==="
echo "epoch5 end (epoch6 start): $(stat -c %y "$RD/diagnostics/epoch_000005.json" 2>/dev/null | cut -c12-19)"
echo "epoch6 selfplay-diag (selfplay END):  $(stat -c %y "$RD/diagnostics/dense_cnn.selfplay.epoch_000006.json" 2>/dev/null | cut -c12-19)"
echo "epoch6 shuffleddata dir (shuffle END): $(ls -dt "$RD/shuffleddata"/*epoch_000006* 2>/dev/null | head -1 | xargs -n1 stat -c %y 2>/dev/null | cut -c12-19)"
echo "epoch6 eval-diag (eval END):           $(stat -c %y "$RD/diagnostics/dense_cnn.evaluation.epoch_000006.json" 2>/dev/null | cut -c12-19)"
echo "epoch6 epoch-json (epoch END):         $(stat -c %y "$RD/diagnostics/epoch_000006.json" 2>/dev/null | cut -c12-19)"
echo ""
echo "=== shuffleddata dirs (one per epoch? growing? = re-shuffle vs incremental) ==="
ls -dt "$RD/shuffleddata"/* 2>/dev/null | head -6 | xargs -n1 sh -c 'echo "$(basename "$0") $(du -sh "$0" 2>/dev/null | cut -f1) npz=$(ls "$0"/*.npz 2>/dev/null | wc -l)"' 2>/dev/null
