DR=/mnt/e/Hexo-BotTrainer/runs
V=/root/.venvs/hexgt-build/bin/python
echo "===== per-epoch diag CONTENT: hexgnn epoch_000004.json vs dense_cnn epoch_000001.json ====="
for r in "hexgnn_rl_main1:epoch_000004" "dense_cnn_rl_main1:epoch_000001"; do
  run=${r%%:*}; ep=${r##*:}
  echo "--- $run/$ep.json keys ---"
  $V -c "import json;d=json.load(open('$DR/$run/diagnostics/$ep.json'));
def keys(o,p=''):
 import collections
 out=[]
 if isinstance(o,dict):
  for k,v in o.items(): out+=keys(v,p+k+'.')
 else: out.append(p[:-1])
 return out
print(sorted(set(keys(d)))[:60])" 2>/dev/null
done
echo ""
echo "===== selfplay diag CONTENT: game-length/diversity stats? ====="
for r in hexgnn_rl_main1 dense_cnn_rl_main1; do
  f=$(ls -t "$DR/$r/diagnostics"/dense_cnn.selfplay.epoch_*.json 2>/dev/null | head -1)
  echo "--- $r: $(basename "$f") ---"
  $V -c "import json;d=json.load(open('$f'));print([k for k in d if not isinstance(d[k],(list,dict))][:30])" 2>/dev/null
  $V -c "import json;d=json.load(open('$f'));print('  has len/diversity/visit keys:',[k for k in d if any(t in k.lower() for t in ('len','divers','visit','prior','decis','entropy','draw','win'))])" 2>/dev/null
done
echo ""
echo "===== epoch_history entry CONTENT (what the loss/eval panels read) ====="
for r in hexgnn_rl_main1 dense_cnn_rl_main1; do
  echo "--- $r epoch_history[last] ---"
  $V - "$r" <<'PY'
import sys,urllib.request,json
d=json.loads(urllib.request.urlopen(f"http://127.0.0.1:8080/api/training/run?name={sys.argv[1]}",timeout=10).read())
eh=d.get('epoch_history',[])
if eh: print('  keys:',sorted(eh[-1].keys()))
PY
done
