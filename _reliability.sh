#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
import glob, torch
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id, AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.graph_build import batch_from_states
from hexo_models.hexgt.losses import decode_binned_value
ck=torch.load('/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/checkpoints/hexgt_rl_epoch000039.pt',map_location='cuda',weights_only=False)
a=ck['arch']
m=HexgtNetwork(token_dim=a['token_dim'],gnn_layers=a['gnn_layers'],ctx_layers=a['ctx_layers'],ffn_dim=a['ffn_dim'],
   attention_heads=a['attention_heads'],short_term_value_horizons=tuple(a.get('short_term_value_horizons',()))).cuda().eval()
m.load_state_dict(ck['model']); torch.set_grad_enabled(False)
def cpn(s): return 'player0' if '0' in str(eng.current_player(s)).lower() else 'player1'
def predv(s):
    b=batch_from_states([s],n=3)
    b={k:(v.cuda() if hasattr(v,'cuda') else v) for k,v in b.items()}
    return float(decode_binned_value(m.forward_policy_value(b)['value'].float().cpu())[0])
# bins
edges=[-1.01,-0.6,-0.2,0.2,0.6,1.01]; lab=['[-1,-.6)','[-.6,-.2)','[-.2,.2)','[.2,.6)','[.6,1]']
import collections
buck=collections.defaultdict(lambda:[0,0.0])  # count, sum_outcome
def run(dirs,k=14):
    files=[]
    for d in dirs: files+=sorted(glob.glob(d+'/*.hxr'))
    done=0
    for f in files:
        if done>=k: break
        rf=HexoRecordFile.open(f); rec=list(rf.iter_records())[0]
        if rec.winner is None: continue
        w=str(rec.winner); coords=[unpack_coord_id(int(c)) for c in rec.action_ids]
        s=eng.new_game()
        for i,c in enumerate(coords[:-1]):  # skip terminal
            cur=cpn(s); pv=predv(s); out=1.0 if cur==w else -1.0
            for bi in range(5):
                if edges[bi]<=pv<edges[bi+1]:
                    buck[lab[bi]][0]+=1; buck[lab[bi]][1]+=out; break
            eng.apply_action(s,PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
        done+=1
    return done
R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
n1=run([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],14)
n2=run([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],14)
print(f"=== Reliability diagram (epoch-39 value head; {n1}+{n2} eval games, per-position) ===")
print(f"{'pred bucket':>10} {'n':>6} {'mean ACTUAL outcome':>20}  (calibrated: actual ~ bucket midpoint)")
for l in lab:
    c,sm=buck[l]
    if c: print(f"{l:>10} {c:>6} {sm/c:>+20.2f}")
PY
