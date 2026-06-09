RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "checkpoints: $(ls "$RD/checkpoints"/epoch_*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
$V -c "import json;d=json.load(open('$RD/diagnostics/dense_cnn.selfplay.live.json'));print('live selfplay epoch',d.get('epoch'),d.get('status'))" 2>/dev/null
for f in $(ls "$RD/diagnostics"/epoch_*.json 2>/dev/null); do
  $V -c "import json;d=json.load(open('$f'));t=d.get('metadata',{}).get('result',{}).get('training',{});print('$(basename $f)','training.status=',t.get('status'),'loss=',t.get('loss'),'components=',list((t.get('loss_components') or {}).keys()))" 2>/dev/null
done
echo "=== 64x10 config STV horizons? ==="
grep -nE "short_term_value_horizons|stvalue|opp_policy_weight|short_term_value_weight" configs/dense_cnn_rl_main1.toml
