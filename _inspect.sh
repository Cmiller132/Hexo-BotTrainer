#!/usr/bin/env bash
python3 - <<'PY'
import json
def load(n): return json.load(open(f"/tmp/{n}.json"))
for run in ("dense_cnn_model1_target_96x8","hexgt_rl_main2"):
    d=load(run)
    print("="*70); print(run)
    h=d.get("histories",[])
    print(" histories:",len(h))
    for e in h[:3]:
        print("   ", {k:e.get(k) for k in ("path","source","name","record_index","winner","turns","label","game_id")})
    # self-play entries specifically
    sp=[e for e in h if "selfplay" in str(e.get("path","")).lower()]
    print(" selfplay histories:",len(sp))
    for e in sp[:3]:
        print("   SP", {k:e.get(k) for k in ("path","source","name","record_index","winner","turns")})
    eh=d.get("epoch_history",[])
    print(" epoch_history[0]:", json.dumps(eh[0]) if eh else None)
    print(" status:", json.dumps(d.get("status")))
PY
