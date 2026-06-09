#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
echo "=== GPU process list (is dense_cnn using it?) ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null | head
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
import glob
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id, AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt import rust_bridge
AX=[(1,0),(0,1),(1,-1)]
def owner_of_move(i):  # move 0 = P0 opening; then turns of 2, starting P1
    if i==0: return 'player0'
    t=(i-1)//2
    return 'player1' if t%2==0 else 'player0'
def hexgt_seat(rf):
    for pl in rf.players:
        if 'hexgt' in str(pl.player_id).lower(): return pl.role
def win_window(coords, winner_cells):
    last=coords[-1]
    for dq,dr in AX:
        for o in range(6):
            win=[(last.q+(k-o)*dq, last.r+(k-o)*dr) for k in range(6)]
            if all((q,r) in winner_cells for q,r in win): return win
    return None
def audit(dirs,label,k=12):
    nblock_in=0; nblock_missing=0; ndouble=0; done=0; csizes=[]
    for d in dirs:
        for f in sorted(glob.glob(d+'/*.hxr')):
            if done>=k: break
            rf=HexoRecordFile.open(f); r=list(rf.iter_records())[0]; seat=hexgt_seat(rf)
            if seat is None or r.winner is None or str(r.winner)==seat: continue  # hexgt losses only
            coords=[unpack_coord_id(int(c)) for c in r.action_ids]
            if len(coords)<8: continue
            owners={}
            for i,c in enumerate(coords): owners[(c.q,c.r)]=owner_of_move(i)
            winner_cells={cell for cell,o in owners.items() if o==str(r.winner)}
            ww=win_window(coords, winner_cells)
            if ww is None: continue
            played=set()
            # walk positions; find LAST hexgt-to-move pos where the win window still had empties
            crit=None
            for i in range(len(coords)):
                if owner_of_move(i)==seat:
                    empties=[(q,r) for (q,r) in ww if (q,r) not in played]
                    winner_in=sum(1 for (q,r) in ww if owners.get((q,r))==str(r.winner) and (q,r) in played)
                    if empties and winner_in>=2:  # threat developing, blockable
                        crit=(i,list(empties),winner_in)
                played.add((coords[i].q,coords[i].r))
            if crit is None: continue
            i,empties,winner_in=crit
            # rebuild state before move i, get candidate set
            s=eng.new_game()
            for j in range(i):
                s=eng.apply_action(s, PlacementAction(AxialCoord(q=int(coords[j].q),r=int(coords[j].r)))) or s
            cand=set()
            for cid in rust_bridge.hexgt_candidate_ids(s,3):
                cc=unpack_coord_id(int(cid)); cand.add((int(cc.q),int(cc.r)))
            csizes.append(len(cand))
            blocks_in=[e for e in empties if e in cand]
            done+=1
            if len(empties)>=2: ndouble+=1   # multiple empties in the winning line at crit (open-ended)
            if blocks_in: nblock_in+=1
            else: nblock_missing+=1
    import statistics as st
    print(f"\n=== {label}: {done} lost games audited ===")
    print(f"  block cell(s) IN candidate set:  {nblock_in}/{done}")
    print(f"  block cell(s) MISSING from cands: {nblock_missing}/{done}  <-- candidate-generation bug if >0")
    print(f"  positions with >=2 open block cells (open-ended/double, harder): {ndouble}/{done}")
    if csizes: print(f"  candidate-set size at crit: med={st.median(csizes):.0f} min={min(csizes)} max={max(csizes)}")
R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
audit([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],'vs dense_cnn',12)
audit([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],'vs SealBot',12)
PY
