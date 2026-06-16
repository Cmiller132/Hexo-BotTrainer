"""Adversarial empirical verification of the HEXGT TSS design doc's threat/defense
claims against the real hexo_engine. Read-only: builds throwaway states.
Run under the WSL CUDA venv (only needs hexo_engine, no torch)."""
from __future__ import annotations

from hexo_engine import api
from hexo_engine.types import PlacementAction, AxialCoord

AX = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}


def popcount(x: int) -> int:
    return bin(x).count("1")


def window_cells(start, axis):
    dq, dr = AX[axis]
    return [AxialCoord(q=start.q + dq * i, r=start.r + dr * i) for i in range(6)]


def analyze(state):
    ps = api.to_python_state(state)
    out = []
    for e in ps.board.windows.entries:
        c0, c1 = popcount(e.masks[0]), popcount(e.masks[1])
        if c0 > 0 and c1 > 0:
            continue  # blocked / not active
        if c0 == 0 and c1 == 0:
            continue
        owner = 0 if c0 > 0 else 1
        count = c0 if owner == 0 else c1
        cells = window_cells(e.key.start, e.key.axis)
        occ = e.masks[0] | e.masks[1]
        empties = [cells[i] for i in range(6) if not (occ >> i) & 1]
        out.append((owner, count, (e.key.start.q, e.key.start.r), e.key.axis,
                    [(c.q, c.r) for c in empties]))
    return out


def threats_for(state, owner, min_count=4):
    return [w for w in analyze(state) if w[0] == owner and w[1] >= min_count]


def play(coords):
    s = api.new_game()
    owners = []
    for (q, r) in coords:
        ps = api.to_python_state(s)
        owners.append(ps.current_player.name[-1])
        res = api.apply_action(s, PlacementAction(coord=AxialCoord(q=q, r=r)))
        if res.terminal:
            print("  [UNEXPECTED terminal mid-build:", api.terminal(s), "]")
    return s, owners


def stm(state):
    ps = api.to_python_state(state)
    return f"{ps.current_player.name}/{ps.phase.name} placements={ps.placements_made}"


def place(state, q, r):
    return api.apply_action(state, PlacementAction(coord=AxialCoord(q=q, r=r)))


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------------------------------------------------------------- TEST B
banner("TEST B: a count-5 window wins with a SINGLE placement (first stone)")
coords = [
    (0, 0),                 # P0 opening
    (0, 7), (2, 7),         # P1
    (1, 0), (2, 0),         # P0 -> 0,1,2
    (4, 7), (6, 7),         # P1
    (3, 0), (4, 0),         # P0 -> 0,1,2,3,4 (count-5 windows live)
    (0, 8), (2, 8),         # P1 (does NOT block the gap) -> P0 to move
]
s, owners = play(coords)
print("owner seq:", "".join(owners), "| state:", stm(s))
fives = [w for w in threats_for(s, 0) if w[1] == 5]
print("P0 count-5 windows:", fives)
gap = None
for w in fives:
    if w[4]:
        gap = w[4][0]
        break
print("placing single stone in gap", gap)
r1 = place(s, gap[0], gap[1])
term = api.terminal(s)
print("  terminal after ONE placement:", r1.terminal, "->", term)
print("RESULT B:", "CONFIRMED count-5 wins in one placement"
      if (term and term.winner.name == 'PLAYER_0') else "REFUTED")

# ---------------------------------------------------------------- TEST C
banner("TEST C: a SIMPLE (single-window) count-4 is neutralized by ONE defender stone\n"
       "        (doc claims a count-4 needs BOTH gaps covered to defend)")
# P0 simple four = stones at offsets 0,1,4,5 of window Q[0..5] (gaps 2,3) + spare = 5 P0 stones.
coords = [
    (0, 0),                 # P0 opening (offset0)
    (0, 7), (2, 7),         # P1
    (1, 0), (4, 0),         # P0 offsets 1,4
    (4, 7), (6, 7),         # P1
    (5, 0), (0, -2),        # P0 offset5 + spare -> simple four {0,1,4,5} gaps{2,3}
    (0, 8), (2, 8),         # P1 -> P0 has the four, now P1 to move
]
s, owners = play(coords)
print("owner seq:", "".join(owners), "| state(P1 to move):", stm(s))
before = threats_for(s, 0)
print("P0 count>=4 windows BEFORE defense:", before)
assert len(before) == 1 and before[0][1] == 4, "expected exactly one simple count-4"
gaps = before[0][4]
print("the four's two gaps:", gaps)
# Defender places ONE stone in ONE gap (other placement wasted far away).
place(s, gaps[0][0], gaps[0][1])      # hit gap #1 only
place(s, 8, 8)                        # waste the second placement
after = threats_for(s, 0)
print("P0 count>=4 windows AFTER one-stone block:", after)
print("RESULT C:", "REFUTES doc: ONE stone neutralizes a simple four"
      if len(after) == 0 else "doc holds for this shape")

# ---------------------------------------------------------------- TEST D (headline)
banner("TEST D: TWO INDEPENDENT simple fours -> doc's set-cover declares HARD LOSS\n"
       "        (union of 4 gaps > 2 placements), but defender survives with 2 stones")
P0 = {0: (0, 0), 3: (1, 0), 4: (4, 0), 7: (5, 0),        # four1 r=0 {0,1,4,5} gaps{(2,0),(3,0)}
      8: (0, 3), 11: (1, 3), 12: (4, 3), 15: (5, 3),     # four2 r=3 {0,1,4,5} gaps{(2,3),(3,3)}
      16: (0, -2)}                                       # spare
P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
      9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
flat = [None] * 17
for i, c in P0.items():
    flat[i] = c
for i, c in P1.items():
    flat[i] = c
s, owners = play(flat)
print("owner seq:", "".join(owners), "| state(P1 to move):", stm(s))
before = threats_for(s, 0)
print("P0 count>=4 windows BEFORE defense:")
for w in before:
    print("   ", w)
allgaps = sorted({g for w in before for g in w[4]})
print(f"doc's set-cover: union of opponent threat gaps = {len(allgaps)} cells {allgaps}")
print(f"  -> doc declares HARD LOSS (2 placements cannot fill {len(allgaps)} gaps)")
# Reality: hitting set of size 2 — one stone in each four.
s_def = s  # P1 to move
place(s_def, 2, 0)   # hit four1
place(s_def, 2, 3)   # hit four2
after = threats_for(s_def, 0)
print("P0 count>=4 windows AFTER P1 plays (2,0),(2,3):", after)
# Confirm P0, now to move, has NO immediate winning completion:
print("  P0 now to move:", stm(s_def))
p0_can_win = len(after) > 0
print("RESULT D:", "REFUTES the depth-1 HARD LOSS override (position is defensible)"
      if not p0_can_win else "doc holds")

# ---------------------------------------------------------------- TEST E
banner("TEST E: an OPEN four (4 consecutive) genuinely needs TWO defender stones\n"
       "        (confirms doc's open-four claim; shows it is the multi-window case)")
coords = [
    (0, 0),                 # P0 opening
    (0, 7), (2, 7),         # P1
    (1, 0), (2, 0),         # P0
    (4, 7), (6, 7),         # P1
    (3, 0), (0, -2),        # P0 -> 0,1,2,3 consecutive (open four) + spare
    (0, 8), (2, 8),         # P1 -> P0 to... wait need P1 to move; check below
]
s, owners = play(coords)
print("owner seq:", "".join(owners), "| state:", stm(s))
before = threats_for(s, 0)
print("P0 count>=4 windows (open four):", before)
# Single block at one end (4,0); waste the other.
s1 = api.clone_state(s)
place(s1, 4, 0)
place(s1, 9, 9)
after1 = threats_for(s1, 0)
print("after blocking ONE end (4,0):", after1, "-> still a live four?" , len(after1) > 0)
# Now show the surviving window is completable -> P0 wins next move (if P0 to move).
print("RESULT E:", "CONFIRMED open four survives a single block (needs 2)"
      if len(after1) > 0 else "single block sufficed (unexpected)")

# ---------------------------------------------------------------- TEST F
banner("TEST F (positive control): TWO independent OPEN fours = genuine forced loss")
P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0),        # open four1 r=0 (0,1,2,3)
      8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3),     # open four2 r=3 (0,1,2,3)
      16: (0, -2)}
P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7),
      9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
flat = [None] * 17
for i, c in P0.items():
    flat[i] = c
for i, c in P1.items():
    flat[i] = c
s, owners = play(flat)
before = threats_for(s, 0)
print("P0 threat windows (two open fours):", len(before), "windows")
threat_window_gapsets = [tuple(sorted(w[4])) for w in before]
print("   distinct threat windows:", len(set(threat_window_gapsets)))
# Best defense attempt: P1 fully blocks open four1 with 2 stones at its ends, then P0
# completes open four2.
place(s, -1, 0)   # block four1 left ext
place(s, 4, 0)    # block four1 right ext (2 stones spent on four1)
rem = threats_for(s, 0)
print("after P1 spends both stones on four1, P0 threats remaining:", len(rem))
# P0 to move: complete four2 (gaps e.g. (-2,3),(-1,3) or (4,3),(5,3))
print("  P0 to move:", stm(s))
won = False
for w in rem:
    if w[1] == 4 and len(w[4]) >= 2:
        g = w[4]
        place(s, g[0][0], g[0][1])
        place(s, g[1][0], g[1][1])
        t = api.terminal(s)
        if t and t.winner.name == 'PLAYER_0':
            won = True
        break
print("RESULT F:", "CONFIRMED genuine forced loss (P0 wins despite P1's best 2 stones)"
      if won else "could not confirm")
