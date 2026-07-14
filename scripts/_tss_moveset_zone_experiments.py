"""CPU experiments for TSS solver defender-move-set trimming (r=2 question).

Mini-model of Hexo rules (verified against hexo_engine bindings):
  - infinite hex grid, axial coords, 3 axes: Q=(1,0), R=(0,1), QR=(1,-1)
  - length-6 windows; win = fill a window entirely with one colour (checked per placement)
  - threat = single-colour window with count >= 4
  - turns: 2 placements per turn (B=2 FirstStone -> B=1 SecondStone); opening ignored
    (experiments start mid-game at a fresh turn)
  - legality: empty cell within hex distance 8 of any stone

Experiments:
  V  : validate mini-model vs real engine on random playouts
  A  : locality stats (win-now / hitting / threat-creating cell distances to nearest stone)
  B1 : random divergence hunt, defender universe FULL vs R2 (per-node radius 2)
  B2 : targeted junction construction (4 count-3 routes crossing at remote cell j)
  C  : candidate-set sizes (FULL vs R2 vs R3) at defender spare nodes
"""
import random, sys, time
from collections import defaultdict

AXES = ((1, 0), (0, 1), (1, -1))
WIN, LOSS, UNK = 1, -1, 0

def hdist(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))

class Mini:
    def __init__(self):
        self.stones = {}          # coord -> 0/1
        self.win = None           # player who completed a window, else None
        self.wcnt = defaultdict(lambda: [0, 0])  # (start, axis_idx) -> [c0, c1]

    def place(self, c, p):
        assert c not in self.stones
        self.stones[c] = p
        delta = []
        won = False
        for ai, ax in enumerate(AXES):
            for off in range(6):
                key = ((c[0] - ax[0] * off, c[1] - ax[1] * off), ai)
                self.wcnt[key][p] += 1
                delta.append(key)
                if self.wcnt[key][p] == 6:
                    won = True
        if won:
            self.win = p
        return (c, p, delta, won)

    def undo(self, delta):
        c, p, keys, won = delta
        del self.stones[c]
        for key in keys:
            self.wcnt[key][p] -= 1
            if self.wcnt[key] == [0, 0]:
                del self.wcnt[key]
        if won:
            self.win = None

    def window_cells(self, key):
        (sq, sr), ai = key
        ax = AXES[ai]
        return [(sq + ax[0] * i, sr + ax[1] * i) for i in range(6)]

    def window_empties(self, key):
        return [c for c in self.window_cells(key) if c not in self.stones]

    def threat_windows(self, p):
        """Single-colour windows with count >= 4 for p."""
        out = []
        for key, cnt in self.wcnt.items():
            if cnt[p] >= 4 and cnt[1 - p] == 0:
                out.append(key)
        return out

def min_hitting_set(sets, budget):
    """Port of threats_shared.rs::min_hitting_set (budget <= 2)."""
    if not sets:
        return 0
    if any(len(s) == 0 for s in sets):
        return None
    universe = []
    for s in sets:
        for c in s:
            if c not in universe:
                universe.append(c)
    if budget >= 1:
        for c in universe:
            if all(c in s for s in sets):
                return 1
    if budget >= 2:
        for i in range(len(universe)):
            for j in range(i + 1, len(universe)):
                a, b = universe[i], universe[j]
                if all(a in s or b in s for s in sets):
                    return 2
    return None

def analyze(st, mover, b):
    """Port of threats_shared.rs::analyze (own_win_now, min_hitting_set)."""
    own_win = False
    opp_sets = []
    opp = 1 - mover
    for key, cnt in st.wcnt.items():
        if cnt[mover] >= 4 and cnt[opp] == 0:
            if cnt[mover] >= 5 or b >= 2:
                own_win = True
        elif cnt[opp] >= 4 and cnt[mover] == 0:
            opp_sets.append(set(st.window_empties(key)))
    return own_win, min_hitting_set(opp_sets, b), len(opp_sets)

def legal_universe(st, patch_r=10):
    """All empties within dist 8 of any stone (bounded to a patch around origin)."""
    out = set()
    for s in st.stones:
        for dq in range(-8, 9):
            for dr in range(-8, 9):
                c = (s[0] + dq, s[1] + dr)
                if c in st.stones or c in out:
                    continue
                if hdist(c, s) <= 8 and hdist(c, (0, 0)) <= patch_r:
                    out.add(c)
    return out

def radius_universe(st, r, patch_r=10):
    out = set()
    for s in st.stones:
        for dq in range(-r, r + 1):
            for dr in range(-r, r + 1):
                c = (s[0] + dq, s[1] + dr)
                if c in st.stones or c in out:
                    continue
                if hdist(c, s) <= r and hdist(c, (0, 0)) <= patch_r:
                    out.add(c)
    return out

def hitting_universe(st, mover):
    """Empties of every opponent >=4 alive window."""
    out = set()
    for key in st.threat_windows(1 - mover):
        out.update(st.window_empties(key))
    return out

def threat_creating(st, mover, universe):
    """Placements creating an own >=4 alive window (window through c with
    own count >= 3 and opp count == 0), ordered by fork degree descending
    (junction-style multi-window moves first -- the H-union-C ordering)."""
    opp = 1 - mover
    out = []
    for c in universe:
        deg = 0
        for ai, ax in enumerate(AXES):
            for off in range(6):
                key = ((c[0] - ax[0] * off, c[1] - ax[1] * off), ai)
                cnt = st.wcnt.get(key)
                if cnt and cnt[mover] >= 3 and cnt[opp] == 0:
                    deg += 1
        if deg:
            out.append((deg, c))
    out.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in out]

class Solver:
    """3-valued depth-limited AND/OR search proving `attacker` wins.

    Attacker nodes: threat-creating moves only; exhausting them yields UNK
    (under-generation can never prove LOSS).
    Defender nodes: lambda1 dispatch; at k == B restrict to hitting universe
    (sound); at k < B use the universe variant under test. Exhausting a
    defender universe claims LOSS -- exactly the property being tested.
    """
    def __init__(self, attacker, variant, depth, node_cap=400_000, patch_r=10,
                 survival_mode=False):
        self.attacker, self.variant, self.depth0 = attacker, variant, depth
        self.node_cap, self.patch_r = node_cap, patch_r
        # survival_mode: defender nodes return UNK at the first unrefuted
        # child (sound for "not proven lost"; do NOT use for LOSS proofs)
        self.survival = survival_mode
        self.nodes = 0
        self.capped = False
        self.memo = {}
        self.escape = None  # (stones_snapshot, move, dist) of a surviving defender move

    def defender_universe(self, st, mover):
        if self.variant == "FULL":
            return legal_universe(st, self.patch_r)
        if self.variant == "R2":
            return radius_universe(st, 2, self.patch_r)
        if self.variant == "R3":
            return radius_universe(st, 3, self.patch_r)
        raise ValueError(self.variant)

    def solve(self, st, mover, b, depth):
        self.nodes += 1
        if self.nodes > self.node_cap:
            self.capped = True
            return UNK
        own_win, k, _n = analyze(st, mover, b)
        if own_win:
            return WIN
        if k is None:
            return LOSS
        if depth == 0:
            return UNK
        key = (frozenset(st.stones.items()), mover, b)
        hit = self.memo.get(key)
        if hit is not None and hit[1] >= depth:
            return hit[0]
        is_att = mover == self.attacker
        if is_att:
            # keep threat_creating's fork-degree ordering (do NOT re-sort)
            moves = threat_creating(st, mover, legal_universe(st, self.patch_r))
        else:
            if k == b and k > 0:
                moves = sorted(hitting_universe(st, mover))
            else:
                moves = sorted(self.defender_universe(st, mover))
        any_unk = False
        result = None
        for c in moves:
            delta = st.place(c, mover)
            if delta[3]:
                r_child_for_mover = WIN
            else:
                nm, nb = (mover, 1) if b == 2 else (1 - mover, 2)
                r = self.solve(st, nm, nb, depth - 1)
                if nm == mover:
                    r_child_for_mover = r
                else:
                    r_child_for_mover = -r if r != UNK else UNK
            st.undo(delta)
            if r_child_for_mover == WIN:
                if not is_att:
                    d = min(hdist(c, s) for s in st.stones)
                    if self.escape is None or d > self.escape[2]:
                        self.escape = (dict(st.stones), c, d)
                result = WIN
                break
            if r_child_for_mover == UNK:
                any_unk = True
                if not is_att:
                    d = min(hdist(c, s) for s in st.stones)
                    if self.escape is None or d > self.escape[2]:
                        self.escape = (dict(st.stones), c, d)
                    if self.survival:
                        result = UNK
                        break
        if result is None:
            if any_unk:
                result = UNK
            elif is_att:
                result = UNK if moves is not None else UNK  # OR under-generation: never LOSS
            else:
                result = LOSS
            if is_att and not any_unk and not moves:
                result = UNK
            if is_att and not any_unk and moves:
                # all threat moves refuted: still only UNK (restricted generator)
                result = UNK
        self.memo[key] = (result, depth)
        return result

# ---------------------------------------------------------------- experiments

def random_position(rng, n_stones=14, bias=0.85):
    """Random legal-ish playout with proximity bias to create line structure."""
    st = Mini()
    st.place((0, 0), 0)
    mover, b = 1, 2
    while len(st.stones) < n_stones:
        own = [c for c, p in st.stones.items() if p == mover]
        if own and rng.random() < bias:
            base = rng.choice(own)
            ax = rng.choice(AXES)
            step = rng.choice([-2, -1, 1, 2])
            c = (base[0] + ax[0] * step, base[1] + ax[1] * step)
        else:
            base = rng.choice(list(st.stones))
            c = (base[0] + rng.randint(-3, 3), base[1] + rng.randint(-3, 3))
        if c in st.stones or hdist(c, (0, 0)) > 8:
            continue
        if min(hdist(c, s) for s in st.stones) > 8:
            continue
        d = st.place(c, mover)
        if d[3]:
            st.undo(d)
            continue
        mover, b = (mover, 1) if b == 2 else (1 - mover, 2)
    return st

def exp_validate(n_games=4, n_plies=40):
    import hexo_engine as he
    def act(q, r):
        return he.PlacementAction(coord=he.AxialCoord(q=q, r=r))
    def unpack(i):
        return ((i >> 16) - 32768, (i & 0xFFFF) - 32768)
    rng = random.Random(7)
    for g in range(n_games):
        eng = he.new_game()
        mini = Mini()
        # opening
        he.apply_action(eng, act(0, 0))
        mini.place((0, 0), 0)
        mover, b = 1, 2
        for ply in range(n_plies):
            ids = he.legal_action_ids(eng)
            if not ids:
                break
            eng_legal = {unpack(i) for i in ids}
            mini_legal = {c for c in legal_universe(mini, patch_r=10**6)}
            assert eng_legal == mini_legal, (
                f"legal mismatch g{g} ply{ply}: "
                f"eng-mini={sorted(eng_legal - mini_legal)[:4]} "
                f"mini-eng={sorted(mini_legal - eng_legal)[:4]}")
            c = unpack(rng.choice(ids))
            res = he.apply_action(eng, act(*c))
            d = mini.place(c, mover)
            assert res.terminal == d[3], \
                f"terminal mismatch g{g} ply{ply}: eng={res.terminal} mini_won={d[3]}"
            if res.terminal:
                break
            mover, b = (mover, 1) if b == 2 else (1 - mover, 2)
    print(f"V : mini-model matches engine on {n_games} random games (legality + terminal)")

def exp_locality(n=400):
    rng = random.Random(11)
    dmax = {"win_now": 0, "hit": 0, "threat_create": 0}
    for _ in range(n):
        st = random_position(rng, n_stones=rng.randint(10, 22))
        stones = list(st.stones)
        for p in (0, 1):
            for key in st.threat_windows(p):
                cnt = st.wcnt[key]
                for c in st.window_empties(key):
                    d = min(hdist(c, s) for s in stones)
                    dmax["win_now" if cnt[p] >= 4 else "hit"] = max(dmax["win_now"], d)
                    dmax["hit"] = max(dmax["hit"], d)
            for c in threat_creating(st, p, legal_universe(st)):
                d = min(hdist(c, s) for s in stones)
                dmax["threat_create"] = max(dmax["threat_create"], d)
    print(f"A : max distance to nearest stone over {n} random positions: "
          f"win-now/hitting cells = {dmax['hit']}, threat-creating = {dmax['threat_create']} "
          f"(theory: <=2 and <=3)")

def exp_sizes(n=200):
    rng = random.Random(13)
    tot = defaultdict(list)
    for _ in range(n):
        st = random_position(rng, n_stones=rng.randint(12, 24))
        tot["FULL"].append(len(legal_universe(st)))
        tot["R2"].append(len(radius_universe(st, 2)))
        tot["R3"].append(len(radius_universe(st, 3)))
    for k, v in tot.items():
        print(f"C : |{k}| mean {sum(v)/len(v):7.1f}  max {max(v)}")

def solve_root(st, attacker, mover, b, variant, depth, node_cap=400_000):
    s = Solver(attacker, variant, depth, node_cap)
    r = s.solve(st, mover, b, depth)
    return r, s

def exp_random_divergence(n=120, depth=7, node_cap=60_000):
    rng = random.Random(17)
    diverge = 0
    checked = 0
    t0 = time.time()
    for i in range(n):
        st = random_position(rng, n_stones=rng.randint(10, 20))
        mover = rng.randint(0, 1)
        # only interesting if attacker has material to attack with
        if not threat_creating(st, mover, legal_universe(st)):
            continue
        checked += 1
        r2, s2 = solve_root(st, mover, mover, 2, "R2", depth, node_cap)
        if r2 != WIN:
            continue
        rf, sf = solve_root(st, mover, mover, 2, "FULL", depth, node_cap)
        if rf != WIN:
            diverge += 1
            esc = sf.escape
            print(f"B1: DIVERGENCE pos {i}: R2=WIN FULL={rf} "
                  f"capped={sf.capped} escape_dist={esc[2] if esc else None}")
    print(f"B1: random hunt: {checked} attackable positions, {diverge} divergences "
          f"({time.time()-t0:.1f}s)")

def fork_geometry_check():
    """Mechanical validation of Codex's G3 defender-counterfork geometry:
    defender arms Q:(8,0)(9,0)(10,0), R:(5,3)(5,4)(5,5), QR:(8,-3)(9,-4)(10,-5);
    playing f=(5,0) creates three defender count-4 alive windows with
    pairwise-disjoint empties => attacker faces min_hit 3 > B=2."""
    st = Mini()
    A, D = 0, 1
    for c in [(8, 0), (9, 0), (10, 0), (5, 3), (5, 4), (5, 5),
              (8, -3), (9, -4), (10, -5)]:
        st.place(c, D)
    # a couple of far attacker stones so the board isn't defender-only
    for c in [(-9, 4), (-8, 4), (-6, 4), (-4, 4)]:
        st.place(c, A)
    f = (5, 0)
    df = min(hdist(f, s) for s in st.stones)
    d = st.place(f, D)
    assert not d[3]
    own_win, k, n = analyze(st, A, 2)  # attacker to move after the fork
    print(f"G3: fork f={f} dist_to_nearest={df} (in R2: {df <= 2}); after fork: "
          f"attacker faces {n} defender threats, min_hit={k} "
          f"(None => lambda1 forced loss for the ATTACKER)")
    st.undo(d)

def junction_position(n_scatter=14, seed=5, caps=True):
    """4 count-3 attacker routes crossing at remote cell j=(0,0), plus a
    SINGLE-WINDOW count-4 attacker pin far away (k=1: defender keeps one
    spare placement). Defender to move (fresh turn, B=2).

    Route arms (attacker stones at offsets 3,4,5 from j):
      +Q: (3,0)(4,0)(5,0)    -Q: (-3,0)(-4,0)(-5,0)
      +R: (0,3)(0,4)(0,5)    -R: (0,-3)(0,-4)(0,-5)
    Pin: stones at QR-line (q+r=6) window (8,-2)..(13,-7), occupying both
    endpoints so exactly ONE window holds 4 of them: q in {8,11,12,13};
    empties (9,-3),(10,-4). The window avoids the route lines r=0 and q=0
    (their crossings with q+r=6 are (6,0) and (0,6), outside the window).
    Defender stones: random far scatter, pairwise sharing no window, in no
    attacker-alive window of count >= 2, all >= 3 from j.
    """
    st = Mini()
    A, D = 0, 1
    arms = [(3, 0), (4, 0), (5, 0), (-3, 0), (-4, 0), (-5, 0),
            (0, 3), (0, 4), (0, 5), (0, -3), (0, -4), (0, -5)]
    pin = [(8, -2), (11, -5), (12, -6), (13, -7)]
    for c in arms + pin:
        d = st.place(c, A)
        assert not d[3]
    if caps:
        # Codex fix: defender caps at outward offset 6 on every arm, so every
        # shifted arm window (e.g. {1..6}, {2..7}, {3..8}) contains either j
        # or a cap. Without them the attacker extends outward after j and
        # wins anyway (the original construction's refuted flaw).
        for c in [(6, 0), (-6, 0), (0, 6), (0, -6)]:
            d = st.place(c, D)
            assert not d[3]
    rng = random.Random(seed)
    placed = 0
    tries = 0
    while placed < n_scatter and tries < 20000:
        tries += 1
        c = (rng.randint(-9, 12), rng.randint(-9, 9))
        if c in st.stones or hdist(c, (0, 0)) < 3:
            continue
        if min(hdist(c, s) for s in st.stones) > 8:
            continue
        bad = False
        for ai, ax in enumerate(AXES):
            for off in range(6):
                key = ((c[0] - ax[0] * off, c[1] - ax[1] * off), ai)
                cnt = st.wcnt.get(key)
                if cnt and (cnt[D] >= 1 or cnt[A] >= 2):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        d = st.place(c, D)
        assert not d[3]
        placed += 1
    assert placed == n_scatter, f"only placed {placed}"
    return st, A, D

def probe(st, prefix, mover, b, attacker, variant, depth, node_cap,
          patch_r=10, survival_mode=False):
    """Apply forced placements (advancing mover/b), then solve."""
    deltas = []
    for c in prefix:
        deltas.append(st.place(c, mover))
        assert not deltas[-1][3]
        mover, b = (mover, 1) if b == 2 else (1 - mover, 2)
    s = Solver(attacker, variant, depth, node_cap, patch_r, survival_mode)
    r = s.solve(st, mover, b, depth)
    for d in reversed(deltas):
        st.undo(d)
    return r, s

def exp_v1_refutation_line():
    """Hand-check of Codex's refutation of the UNCAPPED v1 construction:
    after defender [pin-hit, j], attacker extends outward (6,0)+(-6,0);
    the shifted arm windows avoid j and the defender is lambda-1 lost."""
    st, A, D = junction_position(caps=False)
    seq = [((9, -3), D), ((0, 0), D), ((6, 0), A), ((-6, 0), A)]
    deltas = [st.place(c, p) for c, p in seq]
    own, k, n = analyze(st, D, 2)
    print(f"B2v1: uncapped construction, line [hit,j] vs [(6,0),(-6,0)]: "
          f"defender faces {n} threats, min_hit={k} "
          f"(None => Codex refutation of v1 CONFIRMED)")
    for d in reversed(deltas):
        st.undo(d)

def exp_junction(depth=6, node_cap=400_000):
    st, A, D = junction_position()
    own_win, k, nthreats = analyze(st, D, 2)
    print(f"B2: capped junction: {len(st.stones)} stones, defender to move, "
          f"opp threats={nthreats}, min_hit={k}, own_win_now={own_win}")
    assert nthreats == 1 and k == 1, "construction requires a single k=1 pin"
    j = (0, 0)
    dj = min(hdist(j, s) for s in st.stones)
    print(f"B2: junction cell j={j} dist to nearest stone = {dj} "
          f"(in R2: {dj <= 2}, in R3: {dj <= 3})")
    verd = {WIN: "DEFENDER WINS", LOSS: "ATTACKER WINS (defender proven lost)",
            UNK: "defender survives the horizon (not proven lost)"}
    # Root solve under r2 (expect: complete LOSS proof).
    t0 = time.time()
    s = Solver(A, "R2", depth, node_cap, patch_r=12)
    r = s.solve(st, D, 2, depth)
    print(f"B2: root R2 : {verd[r]:45s} nodes={s.nodes:>8} "
          f"capped={s.capped} ({time.time()-t0:.1f}s)")
    # Survival line: defender-universe restriction is defender-pessimal, so a
    # non-LOSS under R3 soundly implies a non-LOSS under FULL. patch_r=16 so
    # the attacker's pin-line continuations are NOT clipped by the patch.
    hit = (9, -3)
    t0 = time.time()
    r, s = probe(st, [hit, j], D, 2, A, "R3", 8, 600_000,
                 patch_r=16, survival_mode=True)
    print(f"B2: probe [hit pin, play j] (R3, survival): "
          f"{verd[-r if r != UNK else r]:40s} "
          f"nodes={s.nodes} capped={s.capped} ({time.time()-t0:.1f}s)")
    # Exhaustive r2-spare refutation probes.
    r2cells = sorted(radius_universe(st, 2))
    lost, alive = 0, []
    t0 = time.time()
    for spare in r2cells:
        if spare == hit:
            continue
        r, s = probe(st, [hit, spare], D, 2, A, "FULL", 6, 120_000)
        # r is from the attacker's perspective (attacker to move after prefix)
        if r == WIN:
            lost += 1
        else:
            alive.append((spare, r, s.capped))
    print(f"B2: probe [hit pin, r2 spare] over {lost + len(alive)} spares: "
          f"{lost} proven lost, {len(alive)} not proven: {alive[:6]} "
          f"({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "V"):
        exp_validate()
    if which in ("all", "A"):
        exp_locality()
    if which in ("all", "C"):
        exp_sizes()
    if which in ("all", "G3"):
        fork_geometry_check()
    if which in ("all", "V1REF"):
        exp_v1_refutation_line()
    if which in ("all", "B2"):
        exp_junction()
    if which in ("all", "B1"):
        exp_random_divergence()

# ---- closure operator (PROOF doc section 7) + bounded validation ----

def closure_universe(st, defender, D, patch_r=10):
    """R(P, n) per D13 (post-review form): hitting u A-TOUCHED alive-window
    empties u D-alive windows with cnt_D >= 6 - D, intersected with the
    legal set. For D >= 6 there is NO sound dismissal (Z3 boundary):
    return the full legal universe."""
    att = 1 - defender
    legal = legal_universe(st, patch_r)
    if D >= 6:
        return legal
    out = set()
    for key, cnt in st.wcnt.items():
        if cnt[att] >= 1 and cnt[defender] == 0:
            out.update(st.window_empties(key))          # hitting + A-touched
        elif cnt[att] == 0 and cnt[defender] >= 6 - D:
            out.update(st.window_empties(key))          # completion guard
    return out & legal

class SolverZN(Solver):
    """Solver whose defender universe is the closure R(P, n)."""
    def defender_universe(self, st, mover):
        # D = defender placements within remaining horizon; conservative
        D = (self.depth0 // 2) + 1
        return closure_universe(st, mover, D, self.patch_r)

def exp_closure_validation(n=100, depth=6, node_cap=80_000):
    """Bounded model check: ZN must never claim WIN where FULL does not
    (soundness of the closure within horizon `depth`)."""
    rng = random.Random(31)
    checked, diverge, zn_wins = 0, 0, 0
    t0 = time.time()
    for i in range(n):
        st = random_position(rng, n_stones=rng.randint(10, 20))
        mover = rng.randint(0, 1)
        if not threat_creating(st, mover, legal_universe(st)):
            continue
        checked += 1
        s = SolverZN(mover, "ZN", depth, node_cap)
        rz = s.solve(st, mover, 2, depth)
        if rz != WIN:
            continue
        zn_wins += 1
        sf = Solver(mover, "FULL", depth, 400_000)
        rf = sf.solve(st, mover, 2, depth)
        if rf != WIN:
            diverge += 1
            print(f"ZN : DIVERGENCE pos {i}: ZN=WIN FULL={rf} capped={sf.capped}")
    print(f"ZN : closure validation: {checked} positions, {zn_wins} ZN-wins, "
          f"{diverge} divergences ({time.time()-t0:.1f}s)")

def exp_closure_junction(depth=6, node_cap=300_000):
    """The closure must NOT prove the capped-junction position (j is in
    A-alive windows, hence searched)."""
    st, A, D = junction_position()
    s = SolverZN(A, "ZN", depth, node_cap)
    r = s.solve(st, D, 2, depth)
    verd = {WIN: "DEFENDER WINS", LOSS: "ATTACKER WINS", UNK: "not proven (defender survives)"}
    j_in = (0, 0) in closure_universe(st, D, (depth // 2) + 1)
    print(f"ZN : junction: j in closure = {j_in}; root result = {verd[r]} "
          f"nodes={s.nodes} capped={s.capped}")
