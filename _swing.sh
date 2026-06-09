#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
import glob, torch, statistics as st
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id, AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.losses import decode_binned_value

CKPT='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/checkpoints/hexgt_rl_epoch000039.pt'
ck=torch.load(CKPT,map_location='cpu',weights_only=False); a=ck['arch']
m=HexgtNetwork(token_dim=a['token_dim'],gnn_layers=a['gnn_layers'],ctx_layers=a['ctx_layers'],
    ffn_dim=a['ffn_dim'],attention_heads=a['attention_heads'],
    short_term_value_horizons=tuple(a.get('short_term_value_horizons',()))).eval()
m.load_state_dict(ck['model']); torch.set_grad_enabled(False)

def hexgt_seat(rf):
    for pl in rf.players:
        if 'hexgt' in str(pl.player_id).lower(): return pl.role
def val_now(state):  # value from CURRENT player's perspective
    b=batch_from_states([state],n=3); o=m.forward_policy_value(b); return float(decode_binned_value(o['value'])[0])

def analyze_losses(dirs, label, k=8):
    swings=[]; precrash=[]; locs={'open(<25%)':0,'mid(25-60%)':0,'late(>60%)':0}; done=0
    for d in dirs:
        for f in sorted(glob.glob(d+'/*.hxr')):
            if done>=k: break
            rf=HexoRecordFile.open(f); r=list(rf.iter_records())[0]; seat=hexgt_seat(rf)
            if seat is None or r.winner is None or str(r.winner)==seat: continue  # only hexgt LOSSES
            coords=[unpack_coord_id(int(c)) for c in r.action_ids]
            s=eng.new_game(); traj=[]
            for c in coords:
                cp=eng.state_dict(s)['current_player'] if hasattr(eng,'state_dict') else None
                # current player via engine
                pl=eng.current_player(s) if hasattr(eng,'current_player') else None
                v=val_now(s)
                # express in hexgt perspective
                curr = str(pl) if pl is not None else None
                vh = v if (curr is None or curr==seat) else -v
                traj.append(vh)
                eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
            if len(traj)<6: continue
            # swing = first ply after which hexgt value stays <= -0.3
            sw=None
            for i in range(len(traj)):
                if traj[i]<=-0.3 and all(t<=0.0 for t in traj[i:]): sw=i; break
            if sw is None:
                # fallback: biggest single drop
                drops=[(traj[i-1]-traj[i],i) for i in range(1,len(traj))]
                sw=max(drops)[1]
            frac=sw/len(traj)
            swings.append((sw,len(traj),frac))
            precrash.append(max(traj[:sw+1]) if sw>0 else traj[0])
            locs['open(<25%)' if frac<0.25 else ('mid(25-60%)' if frac<0.6 else 'late(>60%)')]+=1
            done+=1
        if done>=k: break
    print(f"\n=== {label}: {done} lost games re-evaluated (epoch-39 net) ===")
    if swings:
        fr=[f for _,_,f in swings]
        print(f"  swing ply: med={st.median([s for s,_,_ in swings]):.0f}  swing/gamelen frac: med={st.median(fr):.2f} mean={st.mean(fr):.2f}")
        print(f"  WHERE it flips to losing: {locs}")
        print(f"  value JUST BEFORE the swing (peak): med={st.median(precrash):+.2f} mean={st.mean(precrash):+.2f}  (>+0.5 = was confidently 'winning' then lost = overconfident)")
        print(f"  # games where hexgt's value was >+0.5 before losing: {sum(1 for p in precrash if p>0.5)}/{len(precrash)}")

R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
analyze_losses([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],'vs dense_cnn losses',8)
analyze_losses([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],'vs SealBot losses',8)
PY
