"""Round-2 verification: phase-awareness (re-confirm G), single-stone block
(re-confirm C), and the hitting-set corollary -- >=3 threat windows can be
defensible with <=2 stones (open four), and a shared common-empty cell is
killed by ONE stone."""
from __future__ import annotations
from itertools import combinations
from hexo_engine import api
from hexo_engine.types import PlacementAction, AxialCoord

AX = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}
def pc(x): return bin(x).count("1")
def wcells(st, ax):
    dq, dr = AX[ax]; return [(st.q + dq * i, st.r + dr * i) for i in range(6)]

def active_threat_windows(state, owner, m=4):
    """Return [(count, [empties as (q,r)])] for active (one-colour) windows with
    count>=m owned by `owner`. An active window = one player's stones, none of the
    other's."""
    ps = api.to_python_state(state); out = []
    for e in ps.board.windows.entries:
        c0, c1 = pc(e.masks[0]), pc(e.masks[1])
        if (c0 > 0) == (c1 > 0):   # both or neither -> not an active single-colour window
            continue
        o = 0 if c0 > 0 else 1; cnt = c0 if o == 0 else c1
        if o == owner and cnt >= m:
            cells = wcells(e.key.start, e.key.axis); occ = e.masks[0] | e.masks[1]
            emp = [cells[i] for i in range(6) if not (occ >> i) & 1]
            out.append((cnt, emp))
    return out

def min_hitting_set(windows, budget=2):
    """Smallest number of cells hitting >=1 empty of every window (<=budget). Returns
    (size or None, the cells). One cell in a window's empties neutralises it."""
    empties = [set(w[1]) for w in windows]
    universe = sorted(set().union(*empties)) if empties else []
    if not empties:
        return 0, []
    for k in range(0, budget + 1):
        for combo in combinations(universe, k):
            cs = set(combo)
            if all(cs & e for e in empties):
                return k, list(combo)
    return None, []   # not hittable within budget

def play(coords):
    s = api.new_game()
    for q, r in coords:
        api.apply_action(s, PlacementAction(coord=AxialCoord(q=q, r=r)))
    return s
def stm(s):
    ps = api.to_python_state(s); return f"{ps.current_player.name}/{ps.phase.name}"
def place(s, q, r): return api.apply_action(s, PlacementAction(coord=AxialCoord(q=q, r=r)))
def banner(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

# ---- re-confirm C: opponent simple four, defender kills it with ONE stone ----
banner("RE-CONFIRM C: one defender stone neutralises a simple count-4")
coords = [(0,0),(0,4),(1,4),(0,2),(0,-2),(4,4),(5,4)]   # P1 four {(0,4),(1,4),(4,4),(5,4)}; P0 to move
s = play(coords)
before = active_threat_windows(s, owner=1)
print("P1 count>=4 windows:", [(c, e) for c, e in before])
hs, cells = min_hitting_set(before, budget=2)
print(f"min hitting set = {hs} cell(s): {cells}")
place(s, before[0][1][0][0], before[0][1][0][1])   # ONE stone in ONE empty
print("after one-stone block, P1 count>=4 windows:", active_threat_windows(s, 1))
print("RESULT C:", "CONFIRMED one stone suffices" if not active_threat_windows(s,1) else "FAIL")

# ---- re-confirm G: at SecondStone a count-4 is NOT win-now ----
banner("RE-CONFIRM G: count-4 is win-now only with TWO placements left (FirstStone)")
coords = [(0,0),(0,7),(2,7),(1,0),(4,0),(4,7),(6,7),(5,0),(0,-2),(0,8),(2,8)]
s = play(coords)
print("FirstStone leaf:", stm(s), "owns count-4:", [(c,e) for c,e in active_threat_windows(s,0)])
place(s, 0, 2)   # waste the FIRST stone on an unrelated cell -> SecondStone
print("now SecondStone leaf:", stm(s), "still owns count-4:", [(c,e) for c,e in active_threat_windows(s,0)])
g = active_threat_windows(s,0)[0][1][0]
r = place(s, g[0], g[1])   # the ONE remaining placement into a gap
print(f"fill one gap {g} with last placement: terminal={r.terminal} -> {api.terminal(s)} | {stm(s)}")
print("RESULT G:", "CONFIRMED count-4 != win-now at SecondStone" if not r.terminal else "FAIL")

# ---- H: open four = THREE threat windows, hitting set = 2 -> DEFENSIBLE ----
banner("RE-CONFIRM/EXTEND H: an open four is 3 threat windows but a 2-cell hitting set\n"
       "         defends it -> '>=3 threat windows = unstoppable' is FALSE")
# P1 open four (4 consecutive) on r=6; P0 to move.  owners: 0 11 00 11 -> P1 has 4 stones.
coords = [(0,0),(0,6),(1,6),(0,2),(0,-2),(2,6),(3,6)]   # P1 = (0,6),(1,6),(2,6),(3,6) open four
s = play(coords)
win = active_threat_windows(s, owner=1)
print(f"P1 threat windows: {len(win)} ->", [(c,e) for c,e in win])
hs, cells = min_hitting_set(win, budget=2)
print(f"min hitting set over {len(win)} windows = {hs} cells: {cells}")
print("=> P0 to move:", stm(s))
for (q, r) in cells:
    place(s, q, r)
print("after P0 plays the 2 hitting cells, P1 count>=4 windows:", active_threat_windows(s, 1))
print("RESULT H:", f"CONFIRMED {len(win)} windows defended by {hs} stones"
      if not active_threat_windows(s,1) and hs is not None and hs <= 2 else "FAIL")

# ---- I: a SHARED common empty cell kills multiple windows with ONE stone ----
banner("EXTEND I: cell X=(0,0) is the shared gap of a Q four AND an R four;\n"
       "         ONE defender stone at X kills BOTH windows (generalises to 3 axes)")
# P0 (opponent) owns Q four {(-2,0),(-1,0),(1,0),(2,0)} gap X=(0,0)&(3,0)
#                 and R four {(0,-2),(0,-1),(0,1),(0,2)} gap X=(0,0)&(0,3).
# 8 P0 stones + 1 spare = 9; P1 to move. Same parity layout as TEST D.
P0 = {0:(-2,0), 3:(-1,0), 4:(1,0), 7:(2,0), 8:(0,-2), 11:(0,-1), 12:(0,1), 15:(0,2), 16:(3,3)}
P1 = {1:(6,0), 2:(8,0), 5:(6,2), 6:(8,2), 9:(6,-2), 10:(8,-2), 13:(6,4), 14:(8,4)}
flat = [None]*17
for i,c in P0.items(): flat[i]=c
for i,c in P1.items(): flat[i]=c
# opening must be (0,0)? No -- opening is forced to (0,0). Adjust: P0 opening is the
# only legal opening, so flat[0] MUST be (0,0). Rebuild so X=(0,0) is NOT P0-owned:
# instead center the cross on a different shared cell, say X=(0,10) (still legal radius).
X = (0, 10)
P0 = {0:(0,0),   # forced opening (a harmless P0 spare, far from the cross)
      3:(-2,10), 4:(-1,10), 7:(1,10), 8:(2,10),       # Q four around X
      11:(0,8), 12:(0,9), 15:(0,11), 16:(0,12)}       # R four around X
P1 = {1:(0,4), 2:(0,5), 5:(3,4), 6:(3,5), 9:(-3,13), 10:(-3,14), 13:(5,8), 14:(5,9)}
flat = [None]*17
for i,c in P0.items(): flat[i]=c
for i,c in P1.items(): flat[i]=c
try:
    s = play(flat)
    win = active_threat_windows(s, owner=0)
    sharing = [w for w in win if X in w[1]]
    print(f"P0 threat windows: {len(win)}; windows whose empties include X={X}: {len(sharing)}")
    for c,e in win: print("   count", c, "empties", e)
    place(s, X[0], X[1])   # ONE stone at the shared cell
    after = active_threat_windows(s, owner=0)
    print(f"after ONE stone at X={X}, P0 count>=4 windows:", after)
    print("RESULT I:", f"CONFIRMED 1 stone killed {len(sharing)} shared-gap windows"
          if len(sharing) >= 2 and not any(X in w[1] for w in after) else "see output")
except Exception as ex:
    print("TEST I construction error:", ex)
