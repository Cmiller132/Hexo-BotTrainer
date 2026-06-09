RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
for ep in 000023 000024; do
  echo "=== epoch $ep ==="
  $V -c "import json;d=json.load(open('$RD/diagnostics/epoch_$ep.json'));r=d['metadata']['result']
sp=r.get('selfplay',{}).get('elapsed_seconds');tr=r.get('training',{});tot=d.get('elapsed_seconds')
print('  TOTAL=%ss | selfplay=%ss | train=%ss (status=%s loss=%s)'%(round(tot or 0),round(sp or 0),round(tr.get('elapsed_seconds') or 0),tr.get('status'),round(tr.get('loss'),2) if tr.get('loss') else None))
sd=r.get('selfplay',{});print('  selfplay: games=%s pos/s=%s'%(sd.get('completed_games'),round(sd.get('positions_per_second') or 0,1)))
ev=r.get('evaluation',{});print('  eval: status=%s games=%s wins=%s mean_turns=%s'%(ev.get('status'),ev.get('games'),ev.get('wins'),ev.get('mean_turns')))" 2>/dev/null
  # mtime split: selfplay-end -> eval-end
  spm=$(stat -c %Y "$RD/diagnostics/dense_cnn.selfplay.epoch_$ep.json" 2>/dev/null)
  evm=$(stat -c %Y "$RD/diagnostics/dense_cnn.evaluation.epoch_$ep.json" 2>/dev/null)
  epm=$(stat -c %Y "$RD/diagnostics/epoch_$ep.json" 2>/dev/null)
  [ -n "$spm" ] && [ -n "$evm" ] && echo "  mtime split: (shuffle+train+eval, selfplay-end->eval-end) = $((evm-spm))s"
  echo "  phase mtimes: selfplay-end=$(stat -c %y "$RD/diagnostics/dense_cnn.selfplay.epoch_$ep.json" 2>/dev/null|cut -c12-19) eval-end=$(stat -c %y "$RD/diagnostics/dense_cnn.evaluation.epoch_$ep.json" 2>/dev/null|cut -c12-19)"
done
echo "=== confirm: how many epochs done + current ==="
ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | wc -l | sed 's/^/checkpoints: /'
ls -t "$RD/checkpoints"/epoch_*.pt 2>/dev/null | head -1 | xargs -n1 basename | sed 's/^/latest: /'
