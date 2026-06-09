#!/usr/bin/env bash
sleep 5  # let the bridge write + the 3s API cache expire
curl -s "localhost:8080/api/training/run?name=hexgt_rl_main2" > /tmp/hx.json
# the actual games-viewer endpoint the history screen fetches:
curl -s "localhost:8080/api/training/history-page?run=hexgt_rl_main2&limit=12" > /tmp/hxhist.json
curl -s "localhost:8080/api/training/history-page?run=dense_cnn_model1_target_96x8&limit=12" > /tmp/dnhist.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/hx.json'))
st=d.get('status',{})
print("=== hexgt_rl_main2 detail (render-relevant) ===")
print("  stage:", st.get('stage'), "| stage_status:", st.get('stage_status'), "| current_epoch:", st.get('current_epoch'))
sl=st.get('selfplay_live',{})
print("  selfplay_live: live=%s epoch=%s games_finished=%s requested=%s"%(sl.get('live'),sl.get('epoch'),sl.get('games_finished'),sl.get('requested_games')))
eh=d.get('evaluation_history',[])
print("  evaluation_history:", [(e.get('epoch'),e.get('wins'),e.get('losses')) for e in eh])
lh=d.get('learning_health',{})
print("  learning_health.status:", lh.get('status'), "| latest_epoch:", lh.get('latest_epoch'))
print("  epoch_history epochs:", [e.get('epoch') for e in d.get('epoch_history',[])])
for tag,f in (("hexgt","/tmp/hxhist.json"),("dense96x8","/tmp/dnhist.json")):
    h=json.load(open(f))
    items=h.get('items',h.get('records',h.get('history',[])))
    print(f"=== history-page [{tag}] ===")
    print("  keys:", list(h.keys()))
    print("  games returned:", len(items) if isinstance(items,list) else items)
    if isinstance(items,list) and items:
        e=items[0]; print("  sample:", {k:e.get(k) for k in ('path','source','winner','game_id','epoch','record_index')})
PY
