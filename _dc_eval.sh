RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
V=/root/.venvs/hexgt-build/bin/python
echo "=== raw grep events.jsonl for eval/sealbot/win/loss ==="
grep -iE "sealbot|eval|win|loss|policy|value" "$RD/diagnostics/events.jsonl" 2>/dev/null | tail -8 | cut -c1-220
echo ""
echo "=== parse events.jsonl: stages + train losses + eval winrate ==="
$V - "$RD/diagnostics/events.jsonl" <<'PY'
import sys,json
for ln in open(sys.argv[1]):
    try: e=json.loads(ln)
    except: continue
    p=e.get('payload',e); st=p.get('stage') or e.get('event','')
    md=p.get('metadata',{}); res=md.get('result',md) if isinstance(md,dict) else {}
    blob=json.dumps(p).lower()
    if any(x in blob for x in ('seal','eval','win','policy_loss','value_loss','"loss"','winrate','train')):
        # extract interesting numbers
        flat={}
        def walk(d,pre=''):
            if isinstance(d,dict):
                for k,vv in d.items(): walk(vv,pre+k+'.')
            elif isinstance(d,(int,float)) and any(t in pre.lower() for t in ('loss','win','seal','rate','epoch','games','draw')):
                flat[pre[:-1]]=d
        walk(p)
        if flat: print(st,'|',{k:flat[k] for k in list(flat)[:8]})
PY
echo ""
echo "=== current phase (rl_train.log or events tail) + any evaluation diag ==="
ls "$RD/diagnostics"/*.json 2>/dev/null | xargs -n1 basename
tail -3 "$(ls -t "$RD"/train.*.out.log 2>/dev/null | head -1)" 2>/dev/null | grep -vE "Warning|frombuffer"
