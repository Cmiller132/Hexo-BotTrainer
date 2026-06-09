#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
import glob, torch
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id, AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.player import HexgtPlayer
AX=[(1,0),(0,1),(1,-1)]
def cpn(s):
    return str(eng.current_player(s)).lower().replace('player_','player').replace('player.','') if False else ('player0' if '0' in str(eng.current_player(s)).lower() else 'player1')
def seat_of(rf):
    for pl in rf.players:
        if 'hexgt' in str(pl.player_id).lower(): return pl.role
def win_window(last, wc):
    for dq,dr in AX:
        for o in range(6):
            w=[(last.q+(k-o)*dq,last.r+(k-o)*dr) for k in range(6)]
            if all(c in wc for c in w): return w
    return None

ck=torch.load('/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/checkpoints/hexgt_rl_epoch000039.pt',map_location='cuda',weights_only=False)
a=ck['arch']
m=HexgtNetwork(token_dim=a['token_dim'],gnn_layers=a['gnn_layers'],ctx_layers=a['ctx_layers'],ffn_dim=a['ffn_dim'],
   attention_heads=a['attention_heads'],short_term_value_horizons=tuple(a.get('short_term_value_horizons',()))).cuda().eval()
m.load_state_dict(ck['model'])
cfg=parse_hexgt_config({"device":"cuda","architecture":{"candidate_radius":3,"short_term_value_horizons":[4,12,24]},
   "selfplay":{"search_visits":512,"c_puct":1.5,"root_policy_temperature":1.0,
               "widening_max_children":96,"widening_min_children":2,"widening_policy_mass":0.95}})
pl=HexgtPlayer(identity_id="hexgt",model=m,config=cfg,device="cuda",fp16=True)
sp=cfg.selfplay
def search_at(state, visits):
    pl.mcts_session.clear()
    r=pl.mcts_session.run([0],[state],pl.inference,visits=visits,c_puct=sp.c_puct,temperature=0.0,seed=7,
        root_policy_temperature=sp.root_policy_temperature,fpu_reduction=sp.fpu_reduction,virtual_loss=sp.virtual_loss,
        widening_policy_mass=sp.widening_policy_mass,widening_max_children=sp.widening_max_children,widening_min_children=sp.widening_min_children)[0]
    cc=unpack_coord_id(r.action_id); return r.root_value,(int(cc.q),int(cc.r))

def get_crit(f):
    rf=HexoRecordFile.open(f); rec=list(rf.iter_records())[0]; seat=seat_of(rf)
    if seat is None or rec.winner is None or str(rec.winner)==seat: return None
    wseat=str(rec.winner); coords=[unpack_coord_id(int(c)) for c in rec.action_ids]
    if len(coords)<8: return None
    s=eng.new_game(); owners=[]
    for c in coords: owners.append(cpn(s)); eng.apply_action(s,PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
    cellmap={(int(coords[i].q),int(coords[i].r)):owners[i] for i in range(len(coords))}
    wc={c for c,o in cellmap.items() if o==wseat}; ww=win_window(coords[-1],wc)
    if ww is None: return None
    s=eng.new_game(); played=set(); crit=None; empt=None
    for i,c in enumerate(coords):
        if owners[i]==seat:
            e=[x for x in ww if x not in played]; wi=sum(1 for x in ww if cellmap.get(x)==wseat and x in played)
            if e and wi>=2: crit=eng.clone_state(s); empt=e
        played.add((int(c.q),int(c.r))); eng.apply_action(s,PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
    if crit is None: return None
    return crit, empt, seat

def run(dirs,label,k=5):
    files=[]
    for d in dirs: files+=sorted(glob.glob(d+'/*.hxr'))
    print(f"\n=== {label}: search-depth on crit (block) positions (epoch-39 net, greedy) ===")
    done=0
    for f in files:
        if done>=k: break
        c=get_crit(f)
        if c is None: continue
        crit,empt,seat=c
        row=[]
        for v in (512,1024,2048,4096):
            rv,mv=search_at(crit,v); blk = mv in empt
            row.append(f"v{v}: val={rv:+.2f} {'BLOCK' if blk else 'no-blk'}")
        print(f"  {f.split('/')[-1]:28s} blocks={empt[:3]} | "+" | ".join(row))
        done+=1

R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
run([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],'vs dense_cnn',5)
run([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],'vs SealBot',5)
PY
