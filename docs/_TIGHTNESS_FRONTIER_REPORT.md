# Tightness frontier report

Date: 2026-07-14.

This report audits the quantitative parameters in the round-6-confirmed T3+
revision of `PROOF_TSS_DEFENDER_ZONES.md`.  `PINNED` means that the displayed
value is attained in the stated framework.  `IMPROVED` means that a smaller
sound value is proved here.  `OPEN` means that neither a full pin nor a sound
improvement is known.

There are two kinds of pin.  An **absolute pin** comes with a weakened
certificate that declares WIN although a legal real line defeats its declared
resolution.  A **relative pin** attains a counting bound or breaks the cited
coupling/coverage lemma, but does not exclude a different proof or certificate
format.  The distinction is recorded for every pin in the final table.

`scripts/_tightness_check.py` verifies the triangle and five-cycle threat
families, selected facts of the T6 counterexample and deadline gadgets,
and isolated distance/count equalities. The remaining prose arguments are
not machine-checked by this script.

```text
triangle: threats=3 tau=2 minimum-witness=3
five-cycle: threats=5 tau=3 minimum-witness=5
T6 own-win omission: K1={k}; omitted d is immediate D completion
deadline roles: OR-COMPLETION and leaf-entry protection checks pass
distance/count chains: rank, virgin arithmetic, touched equality, T5
all tightness checks passed
```

## 1. Frontier table

| ID | Parameter | Current value | Result | Where established |
|---|---|---:|---|---|
| R1a | Exact live-role seed band | `8(r-1)` | **PINNED** | D15, L9′; §2.1 |
| R1b | Uniform live-role seed band | `8(B-1)` | **OPEN** | D11, T4; §2.2 |
| R2 | Virgin-window seed radius | `8(E^D-6)` for `E^D >= 6` | **OPEN** in general; fixed-window arithmetic is attained | D16, L12, §12.2; §3.1 |
| R3 | Touched-window guard | `cnt_D(W)+E^D(W) >= 6` | fixed-window equality attained; full weakened-L12 pin OPEN | D16, L12; §3.2 |
| R4a | LOSS witness cap, `b=1` | 3 | **PINNED** | L13; §4.2 |
| R4b | LOSS witness cap, `b=2` | 6 | **IMPROVED** to 5; 5 is **PINNED** | §4.1 and §4.3 |
| R5a | Internal T6 kernel scope | `mhs <= b` | **PINNED** | T6; §5.1 |
| R5b | Kernel-node not-own-win requirement | T6 premise plus retained D9 diagnostic | PINNED absolutely for combined predicate enforcement; not a single-clause pin | T6; §5.2 |
| R5c | T6 residual threshold | `tau(F \ d) <= b-1` | **PINNED** | T6; §5.3 |
| R6 | Combined LOSS survivor contract | `tau(T) > b` plus the universal-survivor clause | PINNED for the combined contract; not the numeric test alone | D9, T3; §4.4 |
| R7 | D17 substitution transition charge | `Bhat=1+B(C_s)` and `Ehat=1+E(C_s)` | **PINNED** | D17, T9; §6 |
| R8 | D14 local budget recurrence | AND is `1+max`; LOSS is `b` | **PINNED** | D14, L11; §7.1 |
| R9 | Legality-chain step in derived radii | 8 | **PINNED** | D4, L9′, L12; §2.3 |
| R10 | D15 rank/deadline mechanics | AND `+1`, OR `+0`, deadline `0`, max over roles; protect through check | **PINNED** | D10, D15; §§7.2–7.4 |
| R11 | LOSS declared deadline | `leaf-ply+b+2` | **PINNED** | D9, T3; §4.5 |
| R12 | D16 exposure recurrence | AND `1+max`; LOSS `b`; stopping values `0` | **PINNED** | D16, L11; §7.5 |
| R13 | T5 static-cover cutoff | `B <= 3` with `r3` | local radius arithmetic attained; full T5 pin OPEN | T5; §8.1 |
| R14 | L10/T5 short attacker-placement cutoff | first 3 future threat-creating placements | **PINNED** | L10, T5; §8.2 |
| R15 | Independently nonempty AND fallback | at least 1 legal searched reply | Relative/syntactic | D9, T4, D17; §7.6 |
| R16 | Debit used by `B` and `E^D` | every defender placement before the stop | **OPEN** for an `F+H_W` refinement | §12.1 of the proof; §9 |

R4b is the only numerical improvement.  The existing six-window theorem is
sound, but its `K_4` lower example is only an abstract rank-two set system.
Hexo window geometry excludes that example.

## 2. Obligation bands and the legality factor

### 2.1 Exact role radius `8(r-1)` — PINNED

The last sentence after L9′ states the chain arithmetic but does not give a
coordinate trace.  The following trace attains every inequality for rank two.

Let the shared attacker/root stones be

```text
o = (0,0),  z = (0,16).
```

At a defender node `N_0` with budget two, let

```text
x_0 = (8,0),   y = (16,0),
a   = (8,8),   f_0 = (-8,0),   f_1 = (-16,0).
```

The cell `y` carries a live future attacker-placement role.  Its exact ranks
are two at `N_0`, one after the first defender edge, and zero at the deadline.
The ghost later plays `a` and then `y`; `a` is legal through `z`, and `y` is
then legal through `a`, so Z4 holds.

The coupled moves are:

| Defender placement | Real | Ghost | Check |
|---|---|---|---|
| first | `x_0` | `f_0` | `x_0` is legal through `o`; `f_0` is legal through `o` |
| second | `y` | `f_1` | real `y` is newly legal through `x_0`; ghost `y` is still illegal; `f_1` is legal through `f_0` |

The exact distances are

```text
d(x_0,y) = 8 = 8(2-1),
d(y,o) = d(y,z) = 16,
d(a,z) = d(a,y) = 8.
```

Thus radius seven omits the legal seed `x_0`.  The second real defender
placement then makes the protected `y` real-only, while the ghost can still
play its two designated attacker moves.  No transition is terminal: each
defender pair is separated by distance eight and therefore cannot lie in one
length-six window.  The completion guards have exposure below six in this
fragment.

This is a counterexample to the first-protected-occupation conclusion of L9′
under any per-role radius below `8(r-1)`.  For every `r>=2`, the same equality
is realized by taking

```text
y=(8r,0), z=(8r,16), a=(8r,8),
x_i=(8(i+1),0) for 0<=i<r,
f_i=(-8(i+1),0) for 0<=i<r,
```

with `x_(r-1)=y`.  The real defender follows the `x_i` chain, the ghost the
`f_i` chain, interleaved attacker placements stay in a remote shared cluster,
and the final attacker turn plays `a,y`.  Every relay is exactly distance
eight, `y` is initially ghost-illegal, and the first seed is exactly
`8(r-1)` from it.  Choose the initial defender budget to make the `r`th
defender placement end a turn.  The pin is relative: it breaks L9′ and Step
A3's invariant, not a fully specified alternative game theorem.

### 2.2 Uniform radius `8(B-1)` — OPEN

The uniform band substitutes the global local budget `B(N)` for every exact
role rank.  L11 proves `r*(rho) <= B(N)`, so the substitution is sound.  It
does not prove equality for a role that is ghost-illegal and still live.

The rank-two trace in §2.1 pins `r`, not `B`.  A complete D9 certificate that
sets `B=r` must also synchronize the role's deadline with a terminal WIN,
OR-COMPLETION, or LOSS remainder.  Z4 and the turn boundary constrain that
synchronization: a ghost-illegal completion or WIN-witness cell cannot already
have the four or five nearby attacker stones that its terminal use requires,
and a LOSS leaf contributes its remaining `b` placements to `B` after its
leaf-entry deadline.

A pin therefore requires a complete valid certificate with all of the
following properties:

1. a live role has a first real-only occupation after exactly `B` defender
   placements;
2. its first legal dismissed seed is at distance exactly `8(B-1)`;
3. all Z4 attacker legality witnesses, terminal data, LOSS remainder, and D14
   labels are valid; and
4. the seed is outside every other mandatory zone term.

An improvement requires a role-type/turn-tempo theorem of the form
`r <= B-delta` whenever the target is ghost-illegal, or another uniform bound
strictly below `8(B-1)`.  Neither construction nor theorem is present in the
proof or established here.  The sentence after L9′ is therefore a sharpness
statement for the chain inequality, not a pin of the separate `B`-only
wrapper.

### 2.3 Legality coefficient 8 — PINNED

D4 permits a newly legal placement at distance exactly eight.  With only a
stone at `(0,0)`, `(8,0)` is legal and `(16,0)` is illegal; after occupying
`(8,0)`, `(16,0)` becomes legal.  The §2.1 trace uses exactly those two
equalities.  Hence a distance-only L9′ chain cannot replace the coefficient
eight by any smaller coefficient.  The fixed-window trace in §3.1 attains the
same factor on every relay.

The game-rule radius itself is not attackable.  The pin here is relative to
the distance-only legality-chain accounting.  Additional non-distance guards
could avoid a particular distance-eight seed.

## 3. Defender-completion guards

### 3.1 Virgin radius `8(E^D-6)` — OPEN in general

The arithmetic for one fixed window is exact.  Let `E >= 6`, put `k=E-6`,
and let

```text
W = {(i,0) : 0 <= i <= 5},    v = (8,-4).
```

For `k>0`, use relay cells

```text
p_i = -(k-i)v,    0 <= i < k.
```

Put an existing support stone one step from `p_0`; for `k=0`, put it at
`(-1,0)`.  This makes the first placement legal without occupying `W`.
Translate the whole display so that this support stone is the actual occupied
origin.

Then

```text
d(p_0,W) = 8k,
d(p_i,p_(i+1)) = 8,
d(p_(k-1),(0,0)) = 8.
```

After the `k` seed/relay placements, the defender fills the six cells of
`W`.  Exactly `k+6=E` defender placements have occurred.  At `E=6`, the first
placement is `(0,0)` itself.  Choosing initial budget two for even `E` and
one for odd `E` makes the final fill the last placement of a defender turn;
remote attacker turns can be interleaved.  Thus turn parity and immediate
per-placement termination do not improve the fixed-window inequality.  The
`E=7` instance is the sharpness trace recorded after L12 and in §12.2.

This trace does not pin the full verifier term.  `Z_virgin` is a union over
all all-empty windows.  A legal seed lies in 18 incident windows.  If any
incident window has exposure at least six, that window selects the seed at
distance zero even after reducing the contribution from the target `W`.
Blocking the 18 windows with shared stones can legalize later relays; stopping
their clocks with attacker entries creates D10 roles and changes the exposure
labels.  The existing `E=7` sentence does not supply these missing checks.

A general pin needs an unbounded family of exact D9 certificates in which,
at every relay, the cell is outside the complete reduced union

```text
Z_dir union Z_seed union Z_touch union Z_virgin,
```

all other incident virgin windows have exposure below six or are non-D-alive,
and the certificate still resolves with valid turn clocks and Z4 witnesses.
An improvement needs a proof that overlap, attacker-entry timing, or forced
hits always produce a nearer guard.  Neither is known.  The fixed-window
coefficient is attained, but the general zone parameter remains OPEN.

### 3.2 Touched equality -- fixed-window arithmetic attained; full pin OPEN

Let

```text
W = {w_i=(i,0) : 0 <= i <= 5}.
```

At a ghost internal AND node put defender stones at `w_0,w_5`, no attacker
stone in `W`, use defender budget two, and give `W` exact exposure four.  Use
the following coupled defender placements; common attacker placements between
the two defender turns are remote from `W`.

| Defender edge | Real | Ghost |
|---|---|---|
| 1 | `w_1` | `(-5,1)` |
| 2 | `w_2` | `(-3,5)` |
| 3 | `w_3` | `(3,4)` |
| 4 | `w_4` | `(6,-2)` |

Every real fill is legal from an in-window defender stone.  Every displayed
ghost filler is legal from `w_0` or `w_5`; their axis coordinates do not form
a defender threat.  The ghost count in `W` remains two and its
`own_win_now` predicate remains false.  The real counts are

```text
2 -> 3 -> 4 -> 5 -> 6,
```

with immediate defender termination on `w_4`.

At the root the displayed fixed-window count is `2+4=6`. If one stipulates
that the remaining target-window exposure falls to three after the first
ghost edge, the target-window arithmetic does not later recapture `w_1`.
No recurrence-derived terminal certificate is supplied, so this display
makes no claim that every other completion or obligation zone also omits
the real fills. It shows only that neither turn parity nor the raw exposure
arithmetic supports replacing `>=6` by `>6`; it does not by itself falsify
L12.

The budget-one real sequence is

```text
w_1 | A,A | w_2,w_3 | A,A | w_4,w_5.
```

For budget one, the displayed real sequence attains the same fixed-window
equality and completion. This remains an arithmetic trace, not a full
L12 counterexample. A full pin requires a complete D9 certificate with
recurrence-derived exposure, a terminal subtree, searched sets, and all
other zone clauses checked.

## 4. LOSS leaves

### 4.1 Install-ready improvement: sparse witnesses are `3/5`

**L13+ (Hexo-sparse LOSS witnesses). [PROVEN]** A D9 LOSS witness can be
chosen with at most three windows at `b=1` and at most five windows at `b=2`.
Both bounds are sharp among Hexo threat-window families.

*Proof.* Every threat empty set has size one or two.

For `b=1`, the proof in L13 is unchanged.  If the family contains `{a}`, add
one set missing `a`.  Otherwise start with `{a,b}`, add a set missing `a`,
and add a set missing `b`.  At most three sets have transversal number greater
than one.

For `b=2`, first handle a singleton `{a}`. If `{a}` is a member and the
subfamily `H` of members missing `a` had `tau(H)<=1`, then `{a}` together
with an at-most-one-point transversal of `H` would hit the whole family with
at most two points. Thus `tau(H)>1`; the `b=1` selection takes at most three
members of `H`, and adding `{a}` gives at most four, hence at most five.

Assume now that every member has size two.  Choose an inclusion-minimal
subfamily `G` with `tau(G)>2`.  The L13 maximal-disjoint-family proof gives
`|G|<=6`.  Suppose equality holds.  There cannot be three disjoint members,
because those three already have transversal number three.  A maximal
disjoint family cannot have one member, because that two-point member would
hit all of `G`.  Thus it consists of

```text
E_1={a,b},    E_2={c,d}.
```

The four two-point transversals of `E_1,E_2` are `{a,c}`, `{a,d}`, `{b,c}`,
and `{b,d}`.  Each is missed by a member of `G`.  Equality at six requires
four distinct missing members, each missing only its assigned cross-pair;
otherwise the L13 selection uses at most five sets.  A two-set missing
`{a,c}` but meeting the other three cross-pairs must be `{b,d}`.  If it used
only `b` or only `d` from `{a,b,c,d}`, it would miss a second cross-pair; if
it used two outside points, it would be disjoint from both `E_1,E_2`.
Cycling the argument yields `{b,c}`, `{a,d}`, and `{a,c}`.  Therefore a
six-member minimal obstruction is exactly the six edges of `K_4`.

Such a `K_4` is not a Hexo threat-empty family.  A pair that is the empty set
of a threat window is axis-collinear by F1.  Four cells that are pairwise
axis-collinear either lie on one common axis line or are impossible: after
fixing one cell, three different incident axes give coordinates
`(u,0),(0,v),(w,-w)`, and pairwise alignment forces
`u=v=w=-v`, a contradiction.  If three cells already lie on one axis, an
off-line fourth has only two nonparallel axis lines that meet that common
line, so it cannot align with all three distinct cells.  In the common-line
case, order the four empty cells `p_1<p_2<p_3<p_4`.  Any consecutive
length-six window containing `p_1` and `p_3` also contains the intervening
empty `p_2`.  Its empty set therefore cannot be the `K_4` edge
`{p_1,p_3}`.  Contradiction.  Hence `|G|<=5`.  Sharpness is proved in
§§4.2–4.3. ∎

This theorem can replace the `3/6` sentence in D9, L13, and §9 without any
other proof change.  T3 uses only `tau>b`, so the smaller selected family is
drop-in compatible.

### 4.2 Three is sharp at `b=1`

Let

```text
V = {a=(4,0), b=(5,0), c=(4,1)}.
W_ab = {(i,0) : 0 <= i <= 5}.
W_ac = {(4,i) : -4 <= i <= 1}.
W_bc = {(5-i,i) : 0 <= i <= 5}.
```

Put attacker stones on `(W_ab union W_ac union W_bc) \ V` and defender
blockers at

```text
(-1,0), (4,-5), (-1,6).
```

The origin is an attacker stone.  Exhaustive window enumeration gives exactly
three attacker threats, with empty sets

```text
{a,b}, {a,c}, {b,c}.
```

The blockers kill the shifted count-four windows on the three named lines;
every other axis line contains at most three attacker stones.  The family has
transversal number two, while every subfamily of at most two has a common
point.  A `b=1` LOSS leaf therefore requires all three named windows.  After
any one defender placement, one triangle edge is untouched and the attacker
fills its two cells on the following turn.

### 4.3 Five is sharp at `b=2`

Let the five empty vertices be

```text
v_0=(4,0), v_1=(5,0), v_2=(6,0), v_3=(4,2), v_4=(4,1).
```

Use the five windows

```text
W_01 = window((0,0),   (1,0)),
W_12 = window((5,0),   (1,0)),
W_23 = window((6,0),  (-1,1)),
W_34 = window((4,1),   (0,1)),
W_40 = window((4,-4),  (0,1)).
```

Here `window(s,a)={s+i*a:0<=i<=5}`.  Put attacker stones on the union of
these windows minus the five vertices, and put defender blockers at

```text
(-1,0), (4,-5), (11,0), (4,7), (0,6).
```

The origin is an attacker stone.  The complete attacker-threat family is
exactly

```text
{v_0,v_1}, {v_1,v_2}, {v_2,v_3}, {v_3,v_4}, {v_4,v_0}.
```

The finite checker enumerates 20 attacker stones, the five blockers, every
incident window, no complete window, no defender `own_win_now`, and exactly
the five displayed threats.  Their graph is `C_5`, whose vertex-cover number
is three.  Removing any edge leaves a four-edge path with vertex-cover number
two.  Thus no subfamily of at most four has `tau>2`, while all five do.

At a `b=2` LOSS leaf any two defender placements miss a cycle edge.  The
attacker then fills that edge's two legal cells.  The new cap five is therefore
PINNED.  This is a representation-relative pin: a cap four would reject this
valid LOSS leaf, not certify a false WIN.

### 4.4 Combined LOSS survivor contract -- PINNED

The inequality `tau(T)>b` is the finite characterization of D9's
universal survivor clause. Equality is an absolute counterexample to
weakening the LOSS survivor contract itself; it is not a certificate
satisfying the unchanged universal clause. Deleting only the numeric test
is harmless if the universal clause is still verified.

Equality produces a false declared LOSS resolution.  For `b=1`, take

```text
W = {(i,0):0<=i<=5},
A = {(0,0),(1,0),(2,0),(3,0)},
D = {(-1,0)}.
```

The only attacker threat has empty set `{(4,0),(5,0)}` and transversal
number one.  The defender plays `(4,0)`.  The named window is dead, no alive
count-four window remains, and the attacker cannot win in the following two
placements.

For `b=2`, use two copies of this gadget, the second translated by `(20,0)`.
The two disjoint empty pairs have transversal number two.  The defender plays
`(4,0)` and `(24,0)`, killing both.  Again no attacker completion is possible
on the following turn.

Thus equality defeats the combined LOSS survivor contract. It is not an
absolute pin of the numeric test alone while the
universal-survivor clause remains enforced.

### 4.5 LOSS deadline `leaf-ply+b+2` — PINNED

Use `b+1` separated copies of the count-four gadget from §4.4.  Their empty
pairs are disjoint, so the family has transversal number `b+1>b`.  The
defender spends all `b` remaining placements killing `b` different windows.
The surviving window still has exactly two empties.  The first following
attacker placement raises its count from four to five; only the second
completes it.

The resolution therefore occurs exactly `b+2` placements after leaf-ply.
Both the `b` term and the final `+2` are attained.  Any smaller declared
deadline is a false horizon, so this is an absolute pin.

## 5. T6 extendable-hit kernel

### 5.1 Scope `mhs<=b` — PINNED

If `tau(F)>b` and `d` belonged to `K_b`, then `d` plus a residual transversal
of size at most `b-1` would hit all of `F`, contradicting `tau(F)>b`.
Therefore `K_b` is empty already at `mhs=b+1`.

Concrete positions use separated singleton-threat gadgets.  For an empty
cell `e`, take a horizontal window from `e` through `e+(5,0)`, put attacker
stones at offsets one through five, and put defender blockers at offsets
minus one and six.  The gadget has exactly the singleton threat `{e}`.  Two
copies at `b=1` give `tau=2`; three copies at `b=2` give `tau=3`.  The blockers
may be separated so `not own_win_now` holds.  Take `e=(-1,0)` in the first
copy so that its offset-one attacker stone occupies the origin.  In both
cases `K_b` is empty.

This is a grammar/proof-relative pin, not a false-WIN position.  Under
`not own_win_now`, `mhs>b` is a sound LOSS.  T6 already gives the maximal
sound relaxation: stop the internal exact-kernel region and emit a typed LOSS
leaf or hand off to a nonempty sound subtree.  It cannot remain an ordinary
internal AND node that searches exactly an empty kernel.

### 5.2 Kernel-node not-own-win requirement -- combined enforcement PINNED

The following is a complete false-WIN certificate under the combined
weakening described below.
Coordinates are displayed before a common translation by `(-1,0)`; after the
translation an attacker stone occupies the origin.

Let

```text
A_0 = {(0,1),(0,2),(0,3),(1,0),(2,0),(3,0),
       (1,-1),(2,-2),(3,-3),
       (31,10),(32,10),(33,10),(34,10),(35,10),
       (59,25),(66,25)};

D_0 = {(29,10),(36,10),
       (61,25),(62,25),(63,25),(64,25),(65,25)}.

k=(30,10), p=(0,0), d=(60,25).
```

At the root `(A_0,D_0,D,1)`, the complete attacker-threat family is the
single count-five window with empty `{k}`.  Hence `mhs=1=b` and `K_1={k}`.
The defender also has the count-five window

```text
U = {(i,25):60<=i<=65}
```

with sole empty `d`.  The outside attacker stones at `(59,25),(66,25)` block
all shifted defender threats.

The weakened kernel certificate is:

1. The root AND searches exactly `k`; the ghost defender plays `k`.
2. The attacker plays `p`, then `d`.  Both are legal; `d` kills `U`.
3. The resulting defender-budget-two node is a LOSS leaf naming the three
   pairwise-disjoint empty sets

   ```text
   {(-2,0),(-1,0)},
   {(0,-2),(0,-1)},
   {(-2,2),(-1,1)}.
   ```

   Their transversal number is three and the leaf defender has no own win.

All exact transitions and LOSS clauses pass.  In the real game the defender
instead plays omitted `d` at the root and immediately supplies the sixth
stone of `U`.  The real defender wins on the first ply while the restricted
certificate declares attacker WIN.

This position proves that a kernel verifier cannot accept a node with
`own_win_now`. It is an absolute counterexample only to a combined
weakening that deletes the explicit T6 premise and ceases enforcement of
D9's retained internal-AND diagnostic. Deleting the T6 premise alone still
leaves the diagnostic rejecting this root. A replacement kernel may instead
search every immediate Defender completion or reinstate the ordinary
completion zone.

### 5.3 Residual threshold `b-1` — PINNED

After the first defender reply exactly `b-1` placements remain in that turn.
At `b=2`, two disjoint singleton threats have `tau=2`.  Either member of the
minimum transversal leaves one singleton, of residual transversal number one;
these replies must belong to `K_2`.  Replacing `b-1=1` by zero makes the
kernel empty and destroys the followable minimum-transversal line in T6.  At
`b=1`, zero is already the minimum possible residual threshold.  A larger
threshold is sound but searches more cells.  The pin is relative to the exact
kernel construction.

The count ceilings used by T6 are also exact consequences of the premise:
at `b=2` a D-alive window can have three stones and then reach five; at `b=1`
it can have four and then reach five.  Counts four and five respectively
would make `own_win_now` true.  Thus the displayed `3/4/5` constants contain
no slack.

## 6. D17 substitution envelope `+1` — PINNED

The existing proof records both failure modes.  The following coordinates
make their transitions explicit.

For C3, let the shared attacker/root stones be `z=(0,0)` and `u=(-16,16)`, and set

```text
y=(-16,0), d=(-8,0), a=(-16,8), s=(0,1), t=(0,2).
```

At a defender-budget-two node, real `d` is legal through `z`, searched
substitute `s` is legal through `z`, and ghost `y` is illegal because its
distance from both shared stones is 16.  The transitions are:

1. real D plays `d`; ghost D plays `s`;
2. real D plays `y`, newly legal at distance eight from `d`; ghost D plays
   `t`, while ghost `y` remains illegal;
3. shared A plays `a`, legal through `u`, and then designates `y`, now legal
   through `a`.

From `C_s`, one defender placement precedes the `y` deadline, so
`r_C_s(y)=1`.  A child-only radius `8(r-1)=0` permits parent `d`, at distance
eight.  The transition-inclusive radius `8r=8` forbids equality.  Omitting
the `+1` therefore blocks a live obligation in the real game.

For C2, let

```text
W={(i,0):0<=i<=5},
D contains (0,0),(1,0),
d=(2,0), s=(0,1).
```

The current real/ghost transition plays `d/s`.  The remaining synchronized
prefix is:

| Stage | Real | Ghost |
|---|---|---|
| remaining D placement | `(3,0)` | `(0,2)` |
| common A turn | `(0,3),(0,4)` | same |
| next D turn | `(4,0),(5,0)` | `(0,5),(0,6)` |

All placements are legal.  The selected child has three further defender
placements before resolution.  The child-only completion test reads
`2+3=5`; the real count, including the omitted current transition, is

```text
2 -> 3 -> 4 -> 5 -> 6.
```

The transition-inclusive test reads `2+(1+3)=6` and forbids `d`.  D wins
immediately on the last real fill.

Both pins are relative to D17's numerical envelope.  A different verifier
could inspect the current move directly, but it must account for the same C2
and C3 events somewhere.

## 7. Scalar clocks and deadline mechanics

### 7.1 D14 recurrence — PINNED

Let `L(N)` be the maximum number of future defender placements on a
certificate resolution below `N`, including a LOSS remainder.  Finiteness
makes every maximum attainable.  The node grammar gives exactly

```text
L(WIN)=L(OR-COMPLETION)=0,
L(LOSS,b)=b,
L(OR)=L(child),
L(AND)=1+max_C L(C).
```

At an AND node choose a child attaining the finite maximum and then a path
attaining that child's value.  The current edge contributes one defender
placement, so the selected path contains exactly `1+max`.  No smaller scalar
is an upper bound.

The unit term is attained by a defender-budget-one node with two separated
attacker count-five windows.  Search their two last empties.  Either exact
reply leads to a WIN leaf witnessed by the other window.  Each child has
budget zero, but each resolution path contains the current defender edge, so
the parent budget is one.  LOSS examples in §4.5 attain the base value `b`.

D15, D16, and D17 can use smaller role-, window-, or branch-indexed clocks.
They do not improve the scalar quantity D14 defines.  This is a semantic pin
of the recurrence and a relative pin of using that scalar for every guard.

### 7.2 Rank recurrence and maximum over roles — PINNED

An AND edge is one defender placement and an OR edge is none.  The §2.1
rank-two chain fails if either AND increment is omitted.  Zero on an OR edge
is minimal because ranks are nonnegative.  If one cell carries roles of ranks
one and two on different reachable branches, using the minimum or one
arbitrary occurrence repeats the same rank-two failure on the longer branch.
The maximum in D15 is therefore necessary for one parent-wide cell label.
D17 may avoid off-branch roles only by validating a selected branch envelope.

These are relative pins of the ordinary reachable-descendant-union proof.

### 7.3 Rank zero and deadline placement — PINNED

At a role's deadline no defender edge precedes the check, so exact rank zero
is correct and no negative band exists.  At an internal AND node a still-live
exact role necessarily has rank at least one.  The nominal `r=1` radius-zero
seed contribution is extensionally empty: its target is required to be
ghost-illegal, while the candidate set is intersected with legal cells, and a
radius-zero cell can only be the target itself.  An implementation may form
seed bands only for `r>=2` without changing `Z_seed`.

Protection cannot end one defender edge before a leaf-entry check.  Let

```text
W_0={(i,0):0<=i<=5},   W_1={(i,10):0<=i<=5},
```

and put attacker stones at offsets `0,1,4,5` in each.  At a defender-budget-
two root, searched fallback `s=(-1,1)` leads after one defender edge to a
defender-budget-one LOSS leaf.  The two witness empty sets are

```text
{(2,0),(3,0)}, {(2,10),(3,10)},
```

so the ghost leaf has `tau=2>1`.  If the roles are dropped one edge early,
the real defender plays `u=(2,0)` while the ghost plays `s`.  At leaf entry
the real first window is dead.  Its remaining defender placement
`v=(2,10)` kills the second.  No alive count-four window remains, so the real
attacker cannot meet `leaf-ply+1+2`, although the ghost LOSS leaf passes.

Keeping the roles through leaf entry prevents `u`; dropping them immediately
after the check is safe by D9's adaptive remainder.  The deadline is therefore
an absolute pin.

### 7.4 OR-COMPLETION role — PINNED absolutely

Let

```text
W={(i,0):0<=i<=5},
A={(0,0),(1,0),(2,0),(4,0),(5,0)},
c=(3,0).
```

At a defender-budget-one root search fallback `s=(0,1)`.  If D10 omits the
future OR-COMPLETION move `c`, there is no obligation or completion-zone term
that forces `c` into the searched set.  The ghost plays `s`, enters the exact
OR-COMPLETION leaf, and places `c`.  The real defender instead plays dismissed
`c`.  Every alive attacker window with count at least four contains `c`, so
after that reply the real attacker cannot win on the declared ply.  The
weakened certificate is false.

Current D10 protects the one designated cell; the other five cells are shared
attacker stones, exactly as Step O proves.  One cell is both necessary and
sufficient.

### 7.5 D16 exposure recurrence — PINNED

For a fixed D-alive window before its stop, define the actual path exposure as
the number of defender edges before attacker resolution or first attacker
entry into the window.  The exact path maximum obeys D16's recurrence for the
same attained-path reason as D14:

```text
WIN or OR-COMPLETION: 0,
LOSS with remaining budget b: b,
OR entering W: 0,
other OR: child value,
AND: 1+max child value.
```

Each base is attained: a LOSS remainder may use all `b` placements away from
`W`; an ordinary OR consumes no defender placement; and an AND child attaining
the finite maximum gives equality after the current edge.  Setting exposure
to zero once `W` is non-D-alive is exact by permanence.  A branchwise
forced-hit refinement could replace the quantity, but no displayed scalar
clause can decrease while retaining its definition.

### 7.6 Nonempty searched fallback -- PINNED syntactically/relatively

Step A2/A3 needs a legal searched reply to consume a ghost Defender edge
when the real reply is occupied or dismissed. A zero-child AND supplies no
such filler. It is also a maximal node without a typed terminal label, and
the D14/D16 maxima over its children are undefined. Thus nonemptiness is
exact as a syntactic well-formedness and coupling-filler requirement.
Deleting it alone does not admit a false certificate. R15 is a
relative/syntactic pin, not an absolute pin.

## 8. T5 and L10 constants omitted from the question's list

### 8.1 T5 cutoff and radius -- local arithmetic attained; full pin OPEN

For failure at `B=4`, use

```text
W={(-5,0),..., (0,0)},
D={(-1,0),(0,0)},
E^D(W)=4.
```

Take no Attacker stone in any of the 18 windows through `(-5,0)` and no
other current stone within distance three of that cell; put any remaining
certificate data remotely.

The far empty `(-5,0)` is legal and satisfies the local `Z_touch` arithmetic
under the stipulated label `E^D(W)=4`, because `2+4=6`, but its nearest stone
is at distance four.  The window is not attacker-touched, so T5's static set
`r3 union A-touched empties` misses it.

Thus the local static-cover arithmetic cannot extend unchanged to `B=4`.
A full T5 pin additionally requires a D9 certificate realizing the stated
`B` and `E^D` labels; the displayed coordinates alone do not supply it.

At the `B=3` endpoint, three Defender stones at offsets three, four, and
five put the offset-zero empty at distance exactly three while satisfying
the local count equality. Choose no Attacker stone in any window through
the offset-zero cell and put required Attacker/root data remotely, so the
A-touched half of the static union does not select it. Radius two misses it.
This also establishes local arithmetic only until a complete certificate
realizes the stated `B` and exposure labels.

### 8.2 L10 cutoff at three future attacker placements — PINNED

At the current node put attacker stones

```text
{(i,r): i in {0,1,2}, r in {0,1,2}} union {(11,5)}.
```

The origin is occupied.  Let the currently virgin target window be

```text
W={(q,5):0<=q<=5}.
```

The four future attacker moves are

```text
(0,5), (1,5), (2,5), (5,5).
```

For each of the first three, the vertical window at its `q` coordinate
already has three attacker stones at `r=0,1,2`; the placement creates a
count-four threat.  After those moves the fourth placement creates a
count-four threat in `W`, whose three supporting stones are all future stones
relative to the original node.

At the original node `(5,5)` lies in no attacker-touched window.  The anchor
`(11,5)` is distance six, so it makes the cell legal under D4 but shares no
length-six window with it.  Every other current attacker stone differs in
both axial coordinates with positive coordinate sum and is not axis-collinear
with `(5,5)`.  Its nearest current stone is also at distance six, so it is not
in `r3`.  The fourth direct obligation is therefore missed if T5 extends the
L10 conclusion beyond three placements.  The `k<=3` cutoff is relative but
exact for the stated A-touched/static coverage argument.

## 9. Forced-hit debit — OPEN

D14 and D16 count every defender placement before the relevant resolution or
window-entry stop.  Sections 2, 3, and 7 show that these scalar path counts
are exact as definitions.  They do not decide whether a particular window can
use every counted placement as a walk or fill: some placements may be
compulsory hits on attacker threats.

The proposed `F+H_W` debit in §12.1 needs branchwise worst-case bookkeeping
that proves how many compulsory hits are unavailable for filling each chosen
window.  A sound improvement must survive overlapping windows, alternative
minimum transversals, LOSS remainders, and substitutions.  A pin must give a
family in which the defender can attain the present exposure after satisfying
every compulsory hit.  Neither proof exists.  This is distinct from the
fixed-window virgin arithmetic in §3.1 and remains OPEN.

## 10. Inventory boundary

The following numbers are not adjustable T3+ proof parameters:

- window length six and the 18 windows incident to a cell are D2 game
  geometry;
- legality radius eight is D4's game rule;
- D6's count-five/count-four `own_win_now` thresholds follow immediately
  from length six and budgets one/two;
- D16's `cnt_D>=1` touched case and `cnt_D=0` virgin case are the exhaustive
  partition of D-alive windows; the numerical frontier of the latter is R2;
- L1's `6-k`, T1's radius two, and T2's radius three are elementary geometry.

L1 and T2 already state sharp examples.  T1 is sharp with a window whose
attacker stones occupy offsets two through five: offset zero is a legal
two-away completion cell.  The empirical counts in §9 (147/302 cells,
91 positions, 52 claims, and run time) are measurements, not verifier
constants.  T10's finiteness and acyclicity requirements are qualitative;
there is no numerical DAG parameter to tighten.

## 11. Absolute versus framework-relative limit map

| Rows | Pin type | What a smaller value breaks |
|---|---|---|
| R1a, R9 | Relative | L9′ first-protected-occupation and distance-chain accounting |
| R3 | Arithmetic attained; full pin **OPEN** | The fixed-window equality alone does not supply a recurrence-derived D16 certificate or check every other zone |
| R4a, improved R4b | Relative | Completeness of the named-window LOSS leaf format |
| R5a, R5c | Relative | Nonempty exact-kernel grammar and its minimum-transversal line |
| R5b | **Absolute for combined predicate enforcement** | If neither the T6 premise nor the retained D9 diagnostic is enforced, the kernel misses an immediate Defender win |
| R6 | **Absolute for the combined LOSS survivor contract** | Equality permits a complete remainder hitting every witness; deleting only the redundant numeric test does not |
| R7 | Relative | D17/T9 loses the current C2/C3 transition |
| R8, R12 | Relative outside their definitions | The scalar maximum itself is exact; indexed frameworks may avoid using it uniformly |
| R10 rank/max clauses | Relative | D15's ordinary union coupling |
| R10 leaf-entry and OR-COMPLETION clauses | **Absolute** | Concrete weakened certificates declare WIN after the real defender occupies a required cell |
| R11 | **Absolute** | The declared resolution ply is too early |
| R13 | Arithmetic attained; full pin **OPEN** | The local radius trace does not provide a D9 certificate realizing the stated budget and exposure labels |
| R14 | Relative | L10 static coverage |
| R15 | Relative/syntactic | A zero-child AND has no coupling filler and is not defined by the current D9/D14/D16 grammar |

The absolute pins survive any proof framework that retains the same game
outcome and declared horizon: the smaller rule admits a false WIN.  The
relative pins are limits only of the stated coupling, scalar clock, kernel,
or named-witness representation.  They could change under a verifier that
adds different state information, direct current-move tests, branch-indexed
clocks, or a proved forced-hit debit.  The two unresolved numerical frontiers
are the uniform `8(B-1)` wrapper and the full general virgin-window radius;
the `F+H_W` accounting problem is the broader open route that could affect
both completion-zone size and use of scalar budgets.
