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
def cp_name(s):
    cp=eng.current_player(s)
    try: return f'player{cp.index()}'
    except Exception:
        t=str(cp).lower(); return 'player0' if '0' in t else 'player1'
def seat_of(rf):
    for pl in rf.players:
        if 'hexgt' in str(pl.player_id).lower(): return pl.role
def win_window(last, winner_cells):
    for dq,dr in AX:
        for o in range(6):
            w=[(last.q+(k-o)*dq, last.r+(k-o)*dr) for k in range(6)]
            if all(c in winner_cells for c in w): return w
    return None
def owners_seq(coords):
    s=eng.new_game(); owners=[]
    for c in coords:
        owners.append(cp_name(s))
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
    return owners
def audit(dirs,label,k=14):
    nin=nmiss=ndouble=done=nowin=0; csizes=[]
    for d in dirs:
        for f in sorted(glob.glob(d+'/*.hxr')):
            if done>=k: break
            try:
                rf=HexoRecordFile.open(f); rec=list(rf.iter_records())[0]; seat=seat_of(rf)
                if seat is None or rec.winner is None or str(rec.winner)==seat: continue
                wseat=str(rec.winner); coords=[unpack_coord_id(int(c)) for c in rec.action_ids]
                if len(coords)<8: continue
                owners=owners_seq(coords)
                cellmap={(int(coords[i].q),int(coords[i].r)):owners[i] for i in range(len(coords))}
                winner_cells={c for c,o in cellmap.items() if o==wseat}
                ww=win_window(coords[-1], winner_cells)
                if ww is None: nowin+=1; continue
                # pass2: find last hexgt-to-move blockable pos, clone it
                s=eng.new_game(); played=set(); crit_state=None; crit_empties=None
                for i,c in enumerate(coords):
                    if owners[i]==seat:
                        empties=[x for x in ww if x not in played]
                        win_in=sum(1 for x in ww if cellmap.get(x)==wseat and x in played)
                        if empties and win_in>=2:
                            crit_state=eng.clone_state(s); crit_empties=empties
                    played.add((int(c.q),int(c.r)))
                    eng.apply_action(s, PlacementAction(AxialCoord(q=int(c.q),r=int(c.r))))
                if crit_state is None: continue
                cand={(int(unpack_coord_id(int(cid)).q),int(unpack_coord_id(int(cid)).r)) for cid in rust_bridge.candidate_ids(crit_state,3)}
                csizes.append(len(cand)); done+=1
                if len(crit_empties)>=2: ndouble+=1
                if any(e in cand for e in crit_empties): nin+=1
                else: nmiss+=1
            except Exception: pass
    print(f"\n=== {label}: {done} lost games audited (winwin-not-found skipped: {nowin}) ===")
    print(f"  defensive block IN candidate set:   {nin}/{done}")
    print(f"  defensive block MISSING from cands: {nmiss}/{done}   <-- candidate-gen bug if >0")
    print(f"  crit positions open-ended (>=2 block cells): {ndouble}/{done}")
    if csizes: print(f"  candidate-set size: med={st.median(csizes):.0f} min={min(csizes)} max={max(csizes)}")
R='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'
audit([f'{R}/epoch_000039_vs_dense',f'{R}/epoch_000036_vs_dense'],'vs dense_cnn',14)
audit([f'{R}/epoch_000039_vs_sealbot',f'{R}/epoch_000036_vs_sealbot'],'vs SealBot',14)
PY
