RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
$V - "$RD/diagnostics/events.jsonl" <<'PY'
import sys,json
for l in open(sys.argv[1]):
    try:e=json.loads(l)
    except:continue
    p=e.get('payload',{})
    if p.get('stage')=='epoch_000001' and p.get('status')=='completed':
        r=p.get('metadata',{}).get('result',{})
        tr=r.get('train',r.get('training',{}))
        print("epoch0 train:", {k:tr.get(k) for k in tr} if isinstance(tr,dict) else tr)
        # search any loss keys
        import re
        m=re.findall(r'"([a-z_]*loss[a-z_]*)"\s*:\s*([0-9.]+)', json.dumps(r))
        print("loss keys:", m[:8])
PY
