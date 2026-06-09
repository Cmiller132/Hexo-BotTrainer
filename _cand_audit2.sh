#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
import glob, statistics as st
import hexo_engine.api as eng
from hexo_engine.types import unpack_coord_id, AxialCoord, PlacementAction
from hexo_runner.records import HexoRecordFile
from hexo_models.hexgt import rust_bridge
AX=[(1,0),(0,1),(1,-1)]
def owner_of_move(i):
    if i==0: return 'player0'
    return 'player1' if ((i-1)//2)%2==0 else 'player0'
def hexgt_seat(rf):
    for pl in rf.players:
        if 'hexgt' in str(pl.player_id).lower(): return pl.role
def win_window(coords, winner_cells):
    last=coords[-1]
    for dq,dr in AX:
        for o in range(6):
            w=[(last.q+(k-o)*dq, last.r+(k-o)*dr) for k in range(6)]
            if all(c in winner_cells for c in w): return w
    return None
def state_after(coords, upto):
    s=eng.new_game()
    for j in range(upto):
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(coords[j].q), r=int(coords[j].r))))
    return s
def audit(dirs,label,k=12):
    nin=nmiss=ndouble=done=0; csizes=[]
    for d in dirs:
        for f in sorted(glob.glob(d+'/*.hxr')):
            if done>=k: break
            try:
                rf=HexoRecordFile.open(f); rec=list(rf.iter_records())[0]; seat=hexgt_seat(rf)
                if seat is None or rec.winner is None or str(rec.winner)==seat: continue
                wseat=str(rec.winner)
                coords=[unpack_coord_id(int(c)) for c in rec.action_ids]
                if len(coords)<8: continue
                owners={(int(c.q),int(c.r)):owner_of_move(i) for i,c in enumerate(coords)}
                winner_cells={cell for cell,o in owners.items() if o==wseat}
                ww=win_window(coords, winner_cells)
                if ww is None: continue
                played=set(); crit=None
                for i in range(len(coords)):
                    if owner_of_move(i)==seat:
                        empties=[c for c in ww if c not in played]
                        winner_in=sum(1 for c in ww if owners.get(c)==wseat and c in played)
                        if empties and winner_in>=2: crit=(i,empties)
                    played.add((int(coords[i].q),int(coords[i].r)))
                if crit is None: continue
                i,empties=crit
                s=state_after(coords,i)
                cand={(int(unpack_coord_id(int(cid)).q),int(unpack_coord_id(int(cid)).r)) for cid in rust_bridge.hexgt_candidate_ids(s,3)}
                csizes.append(len(cand))
                done+=1
                if len(empties)>=2: ndouble+=1
                if any(e in cand for e in empties): nin+=1
                else: nmiss+=1
            except Exception as ex:
                pass
    print(f"\n=== {label}: {done} lost games audited ===")
    print(f"  defensive block IN candidate set:   {nin}/{done}")
    print(f"  defensive block MISSING from cands: {nmiss}/{done}   <-- candidate-gen bug if >0")
    print(f"  crit positions with >=2 open block cells (open-ended/double): {ndouble}/{done}")
    if csizes: print(f"  candidate-set size: med={st.median(csizes):.0f} min={min(csizes)} max={max(csizes)}")
R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
audit([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],'vs dense_cnn',12)
audit([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],'vs SealBot',12)
PY
