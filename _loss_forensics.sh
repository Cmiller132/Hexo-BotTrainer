#!/usr/bin/env bash
WT=/mnt/e/Hexo-BotTrainer-hexgt
source /root/.venvs/hexgt-build/bin/activate
export PYTHONPATH="$WT/packages/hexo_engine/python:$WT/packages/hexo_utils/python:$WT/packages/hexo_runner/python:$WT/packages/hexo_models/python"
python3 - <<'PY'
from hexo_runner.records import HexoRecordFile
import glob, statistics as st
RUN='/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/eval'

def seat_of_hexgt(rf):
    # rf.players: list of player objects; find the hexgt seat
    for pl in rf.players:
        pid = getattr(pl,'player_id',None) or getattr(pl,'id',None) or str(pl)
        seat = getattr(pl,'role',None)
        if pid and 'hexgt' in str(pid).lower():
            return seat
    return None

def analyze(tag, dirs):
    games=[]  # (hexgt_seat, won(bool), length)
    for d in dirs:
        for f in sorted(glob.glob(d+'/*.hxr')):
            try:
                rf=HexoRecordFile.open(f); recs=list(rf.iter_records())
                if not recs: continue
                r=recs[0]; seat=seat_of_hexgt(rf)
                L=len(list(r.action_ids)); w=r.winner
                if seat is None or w is None:
                    games.append((seat, None, L)); continue
                games.append((seat, (str(w)==seat), L))
            except Exception as e:
                pass
    tot=len(games); wins=sum(1 for _,w,_ in games if w is True); losses=sum(1 for _,w,_ in games if w is False); draws=sum(1 for _,w,_ in games if w is None)
    print(f"\n=== {tag} === games={tot} hexgt W/L/D = {wins}/{losses}/{draws}  winrate={wins/max(1,tot):.1%}")
    # color asymmetry
    for seat in ('player0','player1'):
        g=[x for x in games if x[0]==seat]
        w=sum(1 for _,ww,_ in g if ww is True); l=sum(1 for _,ww,_ in g if ww is False)
        print(f"  as {seat}: {w}W/{l}L  ({'first'if seat=='player0' else 'second'} player)  winrate={w/max(1,w+l):.1%}")
    # length distribution: wins vs losses
    wl=[L for _,w,L in games if w is True]; ll=[L for _,w,L in games if w is False]
    if wl: print(f"  WIN  lengths: n={len(wl)} med={st.median(wl):.0f} mean={st.mean(wl):.1f} min={min(wl)} max={max(wl)}")
    if ll: print(f"  LOSS lengths: n={len(ll)} med={st.median(ll):.0f} mean={st.mean(ll):.1f} min={min(ll)} max={max(ll)}")
    # loss length buckets
    if ll:
        b={'<=20(quick)':0,'21-40':0,'41-60':0,'>60(grind)':0}
        for L in ll:
            if L<=20:b['<=20(quick)']+=1
            elif L<=40:b['21-40']+=1
            elif L<=60:b['41-60']+=1
            else:b['>60(grind)']+=1
        print("  LOSS length buckets:", b)

import os
dense=[f'{RUN}/epoch_{e:06d}_vs_dense' for e in (30,33,36,39,42) if os.path.isdir(f'{RUN}/epoch_{e:06d}_vs_dense')]
seal=[f'{RUN}/epoch_{e:06d}_vs_sealbot' for e in (30,33,36,39,42) if os.path.isdir(f'{RUN}/epoch_{e:06d}_vs_sealbot')]
analyze('vs dense_cnn e24 (epochs 30-42)', dense)
analyze('vs SealBot best-50ms (epochs 30-42)', seal)
PY
