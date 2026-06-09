RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== epoch-0 selfplay diag (train loss? value/policy?) ==="
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.epoch_000001.json'));print({k:d[k] for k in d if any(t in k.lower() for t in ('loss','game','pos','visit','value','win','draw','decis','len','search'))})" 2>/dev/null | head -c 400; echo
echo "=== where does eval write? (dense_cnn evaluation.py) ==="
grep -nE "evaluation|sealbot|win|\.json|diagnostic|write|dump|result" packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/evaluation.py 2>/dev/null | grep -iE "json|write|dump|win_rate|winrate|result|sealbot" | head -6
echo "=== POLL ~6min for eval result (diag file | events | log line) ==="
for i in $(seq 1 30); do
  ev=$(ls "$RD/diagnostics"/*eval* 2>/dev/null)
  hit=$(grep -liE "sealbot.*[0-9]|win.*rate|wins.*[0-9]" "$RD/diagnostics/events.jsonl" "$(ls -t "$RD"/train.*.out.log 2>/dev/null|head -1)" 2>/dev/null)
  ck2=$(ls "$RD/checkpoints"/epoch_000002.pt 2>/dev/null)
  if [ -n "$ev" ] || [ -n "$hit" ] || [ -n "$ck2" ]; then echo "signal: eval_diag=[$ev] hit=[$hit] epoch2_ckpt=[$ck2]"; break; fi
  sleep 12
done
echo "--- eval diag files now ---"; ls "$RD/diagnostics"/ | grep -iE "eval|sealbot" || echo "(none)"
echo "--- grep eval/sealbot/win across diag+log ---"
grep -rhiE "sealbot|win.?rate|[0-9]+W/[0-9]+L|wins.{0,3}[0-9]|win_rate" "$RD/diagnostics/events.jsonl" "$RD"/train.*.out.log 2>/dev/null | tail -6 | cut -c1-200
echo "--- current phase ---"; $V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('selfplay epoch',d.get('epoch'),d.get('status'),'games',d.get('games_finished'))" 2>/dev/null
echo "--- checkpoints ---"; ls "$RD/checkpoints"/*.pt 2>/dev/null | xargs -n1 basename
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
