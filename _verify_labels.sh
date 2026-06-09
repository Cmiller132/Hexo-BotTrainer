#!/usr/bin/env bash
sleep 5  # cache expiry + bridge cycle
curl -s "localhost:8080/api/training/run?name=hexgt_rl_main2" > /tmp/hx.json
curl -s "localhost:8080/api/training/history-page?run=hexgt_rl_main2&limit=400" > /tmp/hxh.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/hx.json')); st=d.get('status',{})
print("HEADER: stage=%s status=%s current_epoch=%s"%(st.get('stage'),st.get('stage_status'),st.get('current_epoch')))
print("epoch_history:", [(e.get('epoch'), 'sp' if e.get('selfplay') else '', 'ev' if e.get('evaluation') else '', 'tr' if e.get('training') else '') for e in d.get('epoch_history',[])])
h=json.load(open('/tmp/hxh.json')); items=h.get('items',[])
# replicate app.js line 2998: item.epoch ? `Epoch ${item.epoch}` : "No epoch"
def label(e): return f"Epoch {e}" if e else "No epoch"
from collections import Counter
sp=[i for i in items if i.get('source')=='selfplay']
ev=[i for i in items if i.get('source')=='evaluation']
print("\nSELF-PLAY games: %d  | labels: %s"%(len(sp), dict(Counter(label(i.get('epoch')) for i in sp))))
print("EVAL games: %d  | labels: %s"%(len(ev), dict(Counter(label(i.get('epoch')) for i in ev))))
noep=[i for i in items if not i.get('epoch')]
print("games with falsy/no-epoch: %d"%len(noep))
# current-epoch count (app.js line 2720): selfplay items where epoch == currentEpoch
ce=st.get('current_epoch')
cur_games=[i for i in sp if i.get('epoch')==ce]
print("current_epoch=%s -> self-play games counted in current epoch: %d"%(ce, len(cur_games)))
# per-epoch selfplay counts
print("self-play games per epoch:", dict(Counter(i.get('epoch') for i in sp)))
PY
