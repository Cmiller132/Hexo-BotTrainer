"""TEST C (fixed parity): opponent owns a SIMPLE four; defender (side-to-move)
neutralizes it with ONE stone -- refuting the doc's 'count-4 needs both gaps
covered' defense rule."""
from __future__ import annotations
from hexo_engine import api
from hexo_engine.types import PlacementAction, AxialCoord
import importlib.util, sys
spec = importlib.util.spec_from_file_location("v", "/mnt/e/Hexo-BotTrainer/scripts/_tss_verify.py")
# reuse helpers without running the module body:
AX = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}
def pc(x): return bin(x).count("1")
def wcells(start, axis):
    dq, dr = AX[axis]; return [AxialCoord(q=start.q+dq*i, r=start.r+dr*i) for i in range(6)]
def analyze(state):
    ps = api.to_python_state(state); out=[]
    for e in ps.board.windows.entries:
        c0,c1=pc(e.masks[0]),pc(e.masks[1])
        if (c0>0)==(c1>0): continue
        owner=0 if c0>0 else 1; count=c0 if owner==0 else c1
        cells=wcells(e.key.start,e.key.axis); occ=e.masks[0]|e.masks[1]
        emp=[(cells[i].q,cells[i].r) for i in range(6) if not (occ>>i)&1]
        out.append((owner,count,(e.key.start.q,e.key.start.r),e.key.axis,emp))
    return out
def threats(state,owner,m=4): return [w for w in analyze(state) if w[0]==owner and w[1]>=m]
def play(coords):
    s=api.new_game(); ow=[]
    for q,r in coords:
        ow.append(api.to_python_state(s).current_player.name[-1])
        api.apply_action(s,PlacementAction(coord=AxialCoord(q=q,r=r)))
    return s,"".join(ow)
def stm(s):
    ps=api.to_python_state(s); return f"{ps.current_player.name}/{ps.phase.name} placements={ps.placements_made}"
def place(s,q,r): return api.apply_action(s,PlacementAction(coord=AxialCoord(q=q,r=r)))

# P1 owns a simple four (4 stones {0,1,4,5} on row r=4); P0 to move (defender).
# owners: 0 11 00 11  -> P1 has 4 stones, then P0 to move.
coords=[(0,0),          # P0 opening (spare)
        (0,4),(1,4),    # P1
        (0,2),(0,-2),   # P0 spares
        (4,4),(5,4)]    # P1 -> four {(0,4),(1,4),(4,4),(5,4)} gaps{(2,4),(3,4)}; P0 to move
s,ow=play(coords)
print("owner seq:",ow,"| state:",stm(s))
before=threats(s,owner=1)
print("P1 count>=4 windows BEFORE defense:",before)
assert len(before)==1 and before[0][1]==4, "want exactly one simple count-4"
gaps=before[0][4]; print("the four's two gaps:",gaps)
# Defender (P0) places ONE stone in ONE gap; wastes the other placement far away.
place(s,gaps[0][0],gaps[0][1])
place(s,1,-2)
after=threats(s,owner=1)
print("P1 count>=4 windows AFTER P0's single-gap block:",after)
print("RESULT C:", "REFUTES doc: ONE defender stone neutralizes a simple four"
      if len(after)==0 else "doc holds (needs 2)")
