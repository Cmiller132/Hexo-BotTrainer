RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
# wait up to ~16 min for epoch 26 to complete (should be ~10-13 min if eval skipped)
for i in $(seq 1 80); do [ -f "$RD/diagnostics/epoch_000026.json" ] && break; sleep 12; done
echo "=== epoch 26 (should SKIP eval) ==="
$V -c "import json,os;d=json.load(open('$RD/diagnostics/epoch_000026.json'));r=d['metadata']['result'];ev=r.get('evaluation',{})
print('epoch 26 TOTAL=%ss (~%.1f min) | train=%s | eval.status=%s eval.reason=%s'%(round(d.get('elapsed_seconds',0)),d.get('elapsed_seconds',0)/60,r.get('training',{}).get('status'),ev.get('status'),ev.get('reason')))
print('eval diag file written for ep26?', os.path.exists('$RD/diagnostics/dense_cnn.evaluation.epoch_000026.json'))" 2>/dev/null
echo "=== compare: eval epochs (24,25) vs skip epoch (26) wall-time ==="
for ep in 000024 000025 000026; do
  $V -c "import json;d=json.load(open('$RD/diagnostics/epoch_$ep.json'));r=d['metadata']['result'];print('epoch $ep: total=%.1f min, eval=%s'%(d.get('elapsed_seconds',0)/60, r.get('evaluation',{}).get('status')))" 2>/dev/null
done
