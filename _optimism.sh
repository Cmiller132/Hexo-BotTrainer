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
from hexo_models.hexgt.graph_build import build_graph_tensors
from hexo_models.hexgt.collate import collate_graphs
from hexo_models.hexgt import rust_bridge
from hexo_models.hexgt.losses import decode_binned_value
ck=torch.load('/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/checkpoints/hexgt_rl_epoch000039.pt',map_location='cuda',weights_only=False)
a=ck['arch']
m=HexgtNetwork(token_dim=a['token_dim'],gnn_layers=a['gnn_layers'],ctx_layers=a['ctx_layers'],ffn_dim=a['ffn_dim'],
   attention_heads=a['attention_heads'],short_term_value_horizons=tuple(a.get('short_term_value_horizons',()))).cuda().eval()
m.load_state_dict(ck['model']); torch.set_grad_enabled(False)
def val_from_facts(facts):
    g=build_graph_tensors(facts); b=collate_graphs([g])
    b={k:(v.cuda() if hasattr(v,'cuda') else v) for k,v in b.items()}
    return float(decode_binned_value(m.forward_policy_value(b)['value'].float().cpu())[0])
def swap_owner(facts):
    own=list(facts['nodes']['node_owner']); sw=[1 if o==0 else (0 if o==1 else o) for o in own]
    return {**facts,'nodes':{**facts['nodes'],'node_owner':sw}}
def cpn(s): return 'player0' if '0' in str(eng.current_player(s)).lower() else 'player1'

files=[]
for ep in (42,41,40): files+=sorted(glob.glob(f'/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_{ep:06d}.hxr'))
# also eval (OOD) for comparison
ev=[]
for d in ('epoch_000039_vs_dense','epoch_000039_vs_sealbot'):
    ev+=sorted(glob.glob(f'/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval/{d}/*.hxr'))

def coords_of(f, multi):
    rf=HexoRecordFile.open(f); recs=list(rf.iter_records())
    return [[unpack_coord_id(int(c)) for c in r.action_ids] for r in recs]

# ---- TEST 1a: same board, both perspectives (FirstStone positions only) ----
def test_a(filelist, label, maxpos=250):
    sums=[]; both_pos=0; n=0
    for f in filelist:
        for coords in coords_of(f, True):
            if n>=maxpos: break
            s=eng.new_game()
            for i,c in enumerate(coords):
                if 10<=i<=50:
                    facts=rust_bridge.graph_facts(s,3)
                    if facts.get('meta',{}).get('first_stone') is None:  # FirstStone -> owner-swap exact
                        vA=val_from_facts(facts); vB=val_from_facts(swap_owner(facts))
                        sums.append(vA+vB); n+=1
                        if vA>0.05 and vB>0.05: both_pos+=1
                        if n>=maxpos: break
                eng.apply_action(s,PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
        if n>=maxpos: break
    if sums:
        print(f"\n[1a] {label}: {n} FirstStone midgame positions (value_A + value_B, identical board)")
        print(f"   mean(vA+vB)={st.mean(sums):+.3f}  median={st.median(sums):+.3f}  (0 = zero-sum-consistent; >0 = optimism)")
        print(f"   BOTH sides predict winning (vA>0 AND vB>0): {both_pos}/{n} = {both_pos/n:.0%}")
        print(f"   sum>+0.5: {sum(1 for x in sums if x>0.5)}/{n}   sum<-0.5: {sum(1 for x in sums if x<-0.5)}/{n}")

# ---- TEST 1b: within-game adjacent cross-player both-positive ----
def test_b(filelist,label,maxg=20):
    pairs=0; bothpos=0; g=0
    for f in filelist:
        for coords in coords_of(f, True):
            if g>=maxg: break
            s=eng.new_game(); seq=[]
            for c in coords:
                seq.append((cpn(s), val_from_facts(rust_bridge.graph_facts(s,3))))
                eng.apply_action(s,PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
            for i in range(len(seq)-1):
                if 10<=i<=50 and seq[i][0]!=seq[i+1][0]:
                    pairs+=1
                    if seq[i][1]>0.05 and seq[i+1][1]>0.05: bothpos+=1
            g+=1
        if g>=maxg: break
    if pairs: print(f"[1b] {label}: adjacent CROSS-player midgame ply-pairs both-positive: {bothpos}/{pairs} = {bothpos/pairs:.0%}")

test_a(files,'SELF-PLAY (in-distribution)',250)
test_a(ev,'EVAL vs dense/sealbot (OOD)',250)
test_b(files,'SELF-PLAY',20)
test_b(ev,'EVAL (OOD)',20)
PY
