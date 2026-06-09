#!/usr/bin/env bash
curl -s "localhost:8080/api/training/run?name=hexgt_rl_main2" > /tmp/hx.json
curl -s "localhost:8080/api/training/run?name=dense_cnn_model1_target_96x8" > /tmp/dn.json
python3 - <<'PY'
import json
def sim(f, label):
    d=json.load(open(f))
    eh=d.get("epoch_history",[])
    live=d.get("status",{})  # /api/training/run returns 'status' (the run status incl current_epoch)
    cur_epoch=live.get("current_epoch")
    def fin(x):
        try: return int(x)
        except: return None
    liveEpoch=fin(cur_epoch)
    seen=[fin(i.get("epoch")) for i in eh]; seen=[x for x in seen if x is not None]
    latest=max(seen) if seen else None
    currentEpoch = liveEpoch if liveEpoch is not None else latest
    prev = currentEpoch-1 if currentEpoch is not None else None
    selfplayEpochs=set(x for x in (currentEpoch,prev) if x is not None and x>=0)
    hs=d.get("histories",[])
    def epoch_of(h):
        p=str(h.get("path","")); import re; m=re.search(r"epoch_(-?\d+)",p); return int(m.group(1)) if m else None
    sp=[h for h in hs if h.get("source")=="selfplay"]
    shown=[h for h in sp if epoch_of(h) in selfplayEpochs]
    print(f"--- {label} ---")
    print(f"  status.current_epoch={cur_epoch}  liveEpoch={liveEpoch}  latestEpochHist={latest}  => currentEpoch={currentEpoch}")
    print(f"  selfplayEpochs={sorted(selfplayEpochs)}")
    print(f"  selfplay histories total={len(sp)}  IN-WINDOW (would render)={len(shown)}")
    print(f"  game epochs present: {sorted(set(epoch_of(h) for h in sp))}")
sim("/tmp/dn.json","dense_96x8 WORKING")
sim("/tmp/hx.json","hexgt_rl_main2")
PY
