# Strategy stealing in engine Hexo, round 3: proxy shadows

**Worktree:** `hunt/gap-raw` at input HEAD `57fcbda8`  
**Date:** 2026-07-17  
**Ranked outcomes:** **(b) achieved for the static class
`C_static^2` [PROVEN]**, and **(c) achieved [PROVEN]**.  
**Global target:** `NL_F` remains **[OPEN]**.

One invented stone of each shadow color does more than repair the count
arithmetic noted in review Finding 6.  The two stones can be scheduled as
genuine legal Hexo placements, the shadow second-player stone can be made an
actual prescription of the allegedly winning strategy, and an isometry can
always be oriented so that the two real first-turn opponent stones avoid both
proxies.  This gives a legal, strategy-consistent role-swapped shadow at the
first synchronized checkpoint and, conditional on successful future placement
transfer, removes the S4 cadence mismatch permanently.

The same construction exposes a sharp limitation.  If those two proxies and
one bijective isometry are then kept fixed, with exact immediate copying in
both directions, at least one proxy has a real preimage that is already legal.
If the simulated strategy's next pair has legal real inverses, the real
opponent can then play that preimage and collide with the proxy; otherwise the
reply lift already failed.  Thus the minimal static exact-copy repair cannot
survive one full ordinary round.  Moving, retiring, backing, or dynamically
recoding proxies remains outside that obstruction.

No Cargo command, Lean build, harness, executable measurement, or production
source edit was used.  This document makes no machine-verified claim.

## 17. Statement boundary and inherited contract

### 17.1 Target, draws, and claim discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real
second player.  The target remains

`NL_F : ∃ pure strategy σ_F, ∀ pure strategies σ_S, S never wins`.

Here “never wins” means either a finite F win or an infinite legal history in
which neither player completes six.  The latter is the round-1/2 meta-level
draw in the declared unbounded idealization, not an engine `GameOutcome`.

Round-2 Theorem D2 is inherited at its exact scope:

`NL_F  ⇔  S has no winning strategy`.                         (D2)

It is **[PROVEN from the CITED Gale–Stewart open-determinacy theorem]** for the
unbounded Hexo macro-game.  Consequently a successful stealing coupling need
only turn an arbitrary alleged winning strategy for S into an F response on
which S cannot win.  D2 does not select that alternative by itself, so
`NL_F` is still **[OPEN]**.

Every named or load-bearing proposition below is marked **PROVEN**,
**SKETCH**, **CONJECTURE**, **OPEN**, or **CITED**.  Stipulative definitions
are labeled `Definition`.  Explanatory prose inherits the status and
hypotheses of the result it follows.

### 17.2 Round-1/2 rules, unchanged [PROVEN]

This round uses the previous formalization verbatim and introduces no rule
variant.  Let

`H = {(q,r): q,r ∈ Z}`

with

`d((q,r),(q',r')) = max(|q-q'|, |r-r'|, |(q-q')+(r-r')|)`.

For finite nonempty occupancy `O`, put

`N_8(O)={c∈H : min_{z∈O} d(c,z)≤8}`, and `L(O)=N_8(O)\O`.

The only opening is `F@(0,0)`.  Along nonterminal play the owner cadence is

`F ; S,S ; F,F ; S,S ; F,F ; …`.

A normal placement is legal exactly when it is empty and lies in `L(O)`.
After a nonwinning first placement the legal store is updated before the
second placement, so an ordered normal turn `(c_1,c_2)` is legal exactly
when

`c_1∈L(O)`, `c_1` is nonwinning, and
`c_2∈L(O∪{c_1})`.

Either owner wins immediately after occupying all six cells of a window in
one of the directions `(1,0)`, `(0,1)`, `(1,-1)`; a first-placement win
suppresses the second placement.

These facts are implemented by
`packages/hexo_engine/rust/src/coord.rs:1-4,9-20,76-95`,
`packages/hexo_engine/rust/src/board.rs:1-5,91-95,167-170`,
`packages/hexo_engine/rust/src/legal.rs:17-18,123-145`,
`packages/hexo_engine/rust/src/rules.rs:16-44`,
`packages/hexo_engine/rust/src/state.rs:149-160,289-357`, and
`packages/hexo_engine/rust/src/tactics.rs:13-17,21-75,205-208,318-333,451-485`.
The literal coordinate carrier is `i16`; the general theorems concern the
declared `Z²` idealization, while every explicit finite gadget below is
checked inside a small safe region.

Round-2 D6 equivariance is also inherited **[PROVEN]**.  Each
`g∈D6` fixes the origin and preserves distance, legal growth, the three
unoriented window directions, owner, phase, and outcome.  A map
`T(c)=t+g(c)` is an isometry for normal play; its translation is not a
full-game symmetry because the actual opening is rooted at the origin.

### 17.3 Standing negative results, at their exact scopes [PROVEN]

The following results are neither weakened nor inflated here.

- Round-1 S3 proves that identity/deletion is not legality-monotone: on its
  exact legal witness, deleting real `F@(0,0)` makes real `S@(8,0)`
  unsupported.  It does not exclude translation, recoding, lag, or proxies.
- Round-1 S4 proves that deleting only the real opening cannot turn
  `F;(S,S)` into a legal role-swapped full-game prefix.  It is a cadence
  obstruction to that one-deletion construction, not an outcome theorem.
- Round-2 S9 proves the equal-positive-odd omission law at a legal
  nonterminal same-actor `FirstStone` checkpoint.  S9.1 consequently
  excludes synchronous, owner-faithful, **no-invention**, injectively
  one-for-one total shadows there, for every coordinate map.  The
  no-invention premise is load-bearing.
- Round-2 S11 excludes `C_iso`: universal, same-phase, owner-faithful,
  no-invention, role-swapped subset shadows at the covered real S
  `FirstStone` checkpoints, using one history-dependent translation/D6
  isometry and immediately copying the next placement.  It does not cover
  the proxy class defined below.

Round-2 S5–S7 remain the governing surplus calculus **[PROVEN]**.  In
particular, for `A⊆A∪E`, the real-only frontier is

`Γ(A,E)=(N_8(E)\N_8(A))\(A∪E)`.                            (17.1)

The formula is color-blind; collision with an occupied coordinate is a
separate issue.

## 18. Proxy-coupling model and proof obligations

### 18.1 Genuine and virtual proxies

**Definition 18.1 (role-swapped proxy shadow).**  The shadow opener
`Ŝ` represents real S, and the shadow second player `F̂` represents real
F and is controlled by an alleged second-player strategy `σ`.  At a
normalized live checkpoint let

- `A` be the finite occupied coordinates represented on both boards, with
  owner fidelity under the role swap;
- `E` be real-only surplus coordinates; and
- `P=P_Ŝ∪P_F̂` be invented shadow-only proxy coordinates.

Thus the normalized real occupancy is `A∪E`, while the shadow occupancy is
`A∪P`.  A budget `(β_Ŝ,β_F̂)` requires
`|P_Ŝ|≤β_Ŝ` and `|P_F̂|≤β_F̂` at every live checkpoint.  A coordinate map
may be used to reach this normalization; in the positive construction it is
a translation followed by an element of D6.

A **genuine-game proxy** is invented only relative to the real board: it is
still an actual placement in one legal shadow Hexo history, at the correct
owner and phase.  A **virtual-board proxy** is inserted without such a legal
shadow placement.  In the latter model `σ` is not automatically defined,
because a winning strategy is promised only on legal histories.

**Domain boundary [PROVEN].**  Querying `σ` after genuine-game proxies is
legitimate when the entire shadow prefix is legal and its earlier
`F̂` moves agree with `σ`.  Merely inserting virtual-board proxies does
not establish either fact.  Any virtual-board construction therefore owes a
separate strategy-domain theorem.  This follows from the definition of a
pure strategy on the legal game tree; it is not an additional engine rule.

### 18.2 Coupling data

**Definition 18.2 (live proxy coupling).**  For a fixed alleged winning
strategy `σ`, a live coupling supplies after each covered real prefix:

1. a legal, nonterminal shadow prefix;
2. bounded proxy sets and an owner-faithful representation of every declared
   common stone;
3. a causal history map: its choices depend only on the observed real prefix
   and on prescriptions already returned by `σ`;
4. an explicit relation between real and shadow `FirstStone`/
   `SecondStone` phases;
5. a rule for appending each real S placement as a shadow `Ŝ` placement;
6. a rule for realizing each sequential `σ`-prescribed shadow `F̂`
   placement as a legal real F placement; and
7. a terminal rule sufficient to show that a real S win would contradict
   `σ`, while a shadow `F̂` win yields a real F win or otherwise prevents
   S from winning.

The coupling may stop successfully on a finite real F win or on a legal
shadow `Ŝ` win, since the latter contradicts the premise that `σ` wins
every shadow play.  It may not silently continue through a terminal shadow
state.

### 18.3 Initial obligation ledger

| Obligation | Status before the new theorems | Exact issue |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **OPEN** for virtual proxies | The shadow queried by `σ` must be a legal history consistent with its own past prescriptions |
| `P1 OPENING/CADENCE` | **OPEN** here, discharged in §19 | Account for the real opening and both first-turn S stones without S4's count mismatch |
| `P2 REAL→SHADOW` | **OPEN** globally | Map every legal real S placement, including moves supported only by real surplus `E` |
| `P3 SHADOW→REAL` | **OPEN** globally | Lift both sequential `σ` replies even when a proxy supplies their only shadow support |
| `P4 COLLISION` | **OPEN** globally | Handle a copied coordinate already occupied by a proxy or real surplus |
| `P5 TERMINAL` | **OPEN** globally | Prevent or harmlessly interpret a six using an invented stone, including a second-placement win |
| `P6 CAUSALITY` | **OPEN** globally | Do not expose a fixed future real F coordinate across an intervening S turn when S12's subset, legality, and real-emptiness hypotheses hold |

Sections 19–22 discharge `P1`, give exact set formulas for `P2`–`P4`,
and prove that the natural static two-proxy choice fails.  They do not
discharge the global `P3`–`P6` conjunction.

## 19. Ranked outcome (c): two genuine proxies synchronize the opening

### 19.1 Strategy-consistent construction

Take an arbitrary legal real opening prefix

`F@x ; S@a,S@b`, with `x=(0,0)`.                         (19.1)

Legality gives

`a,b≠x`, `a≠b`, `d(a,x)≤8`, and
`min(d(b,x),d(b,a))≤8`.                                  (19.2)

Let `σ` be any legal pure strategy for the original second-player role; in
the stealing application it is the alleged winning strategy of real S.

**Theorem S15 (strategy-consistent two-proxy opening synchronization)
[PROVEN].**  There is a legal role-swapped shadow prefix with exactly one
invented stone of each shadow color,

`Ŝ@p_S ; F̂@q_1,F̂@q_2 ; Ŝ@T(a),Ŝ@T(b)`,                 (19.3)

such that

1. `p_S=(0,0)` is the invented shadow-`Ŝ` opening proxy;
2. `q_1,q_2` are exactly `σ`'s sequential first-turn prescriptions;
3. `q_1` is the invented shadow-`F̂` proxy;
4. `q_2=T(x)` represents the compulsory real F opening;
5. `T(c)=q_2+g(c)` for some `g∈D6`; and
6. after (19.3), both histories are nonterminal with real F/shadow `F̂`
   to act at `FirstStone`.

*Proof.*  Place the shadow-`Ŝ` proxy at the compulsory origin and query
`σ` sequentially.  Write its legal first and second coordinates as
`q_1,q_2`.  The first cannot win with one `F̂` stone, so the second
prescription is reached.  The second cannot win with only two `F̂` stones.
Thus

`Ŝ@(0,0); F̂@q_1,F̂@q_2`

is a legal, nonterminal shadow prefix consistent with `σ`.  Declare
`q_1` to be the `F̂` proxy and `q_2` to represent real `x`.

It remains to choose the orientation `g`.  For
`c∈{a,b}` and `p∈{0,q_1}`, a collision `T(c)=p` is exactly

`g(c)=p-q_2`.                                             (19.4)

The group D6 has twelve elements.  A nonzero lattice vector is fixed by at
most one nonidentity D6 reflection and by no nonidentity rotation, so its
stabilizer has size at most two.  Consequently each equation in (19.4)
excludes at most two choices of `g`.  There are four equations, so their
union excludes at most eight of the twelve choices.  Choose any remaining
`g`.

This choice makes `T(a),T(b)` avoid both proxies.  Neither equals
`q_2=T(x)`, because `a,b≠x`; and they are distinct because `T` is
injective.  Equation (19.2) and isometry give

`d(T(a),q_2)≤8`,

and

`min(d(T(b),q_2),d(T(b),T(a)))≤8`.

Hence `T(a)` is a legal shadow first placement and `T(b)` is a legal
shadow second placement after it.  Shadow `Ŝ` then owns only three stones
and shadow `F̂` only two, so no terminal window exists.  The shadow counts
are

`(|X_Ŝ|,|X_F̂|)=(3,2)`,

the legal first normal `Ŝ`-then-`F̂` checkpoint with `F̂` to move.
The real counts are `(|X_F|,|X_S|)=(1,2)`, also with F to move at
`FirstStone`.

The construction is causal.  The pair `q_1,q_2` depends only on the fixed
strategy and the forced shadow opening.  The orientation `g` is selected
only after the already-completed real S pair is observed, and all earlier
represented/proxy coordinates remain fixed because `T(x)=q_2` for every
`g`.

Finally, with `||c||_h=d(c,0)`, legality gives
`||q_1||_h≤8`, `||q_2||_h≤16`, `||a||_h≤8`, and
`||b||_h≤16`.  Thus `||T(a)||_h≤24` and
`||T(b)||_h≤32`.  The radius-eight legal-update halo therefore has norm at
most 40.  All coordinates enumerated for (19.3) are inside the same safe
`i16` region used in rounds 1–2. ∎

The theorem uses genuine-game proxies, not a virtual board.  In particular,
it discharges `P0 STRATEGY-DOMAIN` for this prefix **[PROVEN]**: the first
shadow `F̂` pair really is the pair prescribed by `σ`, and both subsequent
shadow-`Ŝ` placements are legal shadow-opponent choices in that same shadow
game.

### 19.2 Cadence never mismatches again

**Corollary S15.1 (permanent cadence repair, conditional only on successful
future placement transfer) [PROVEN].**  Starting from S15, suppose every later
nonterminal real placement is matched by exactly one same-role shadow
placement, with the two proxies retained.  Then real and shadow
`FirstStone`/`SecondStone` phases remain identical at every later live
checkpoint, and the shadow retains an offset of exactly one stone of each
role.  No later step can fail merely because of singleton-versus-pair count
arithmetic.

*Proof.*  At the real F `FirstStone` checkpoint after S completes its
`k`-th normal turn, `k≥1`, the real counts are

`(|X_F|,|X_S|)=(2k-1,2k)`.

Mapping every real stone and adding the two proxies gives shadow-role counts

`(|X_Ŝ|,|X_F̂|)=(2k+1,2k)`,                              (19.5)

which is exactly a legal shadow-`F̂` `FirstStone` checkpoint.  At the real
S `FirstStone` checkpoint after F completes its `k`-th normal turn, the
real counts are `(2k+1,2k)`, and the shadow-role counts are

`(|X_Ŝ|,|X_F̂|)=(2k+1,2k+2)`,                            (19.6)

exactly a legal shadow-`Ŝ` `FirstStone` checkpoint.  Between these
checkpoints, one successful first-placement correspondence changes both
phases to `SecondStone`, and one successful nonwinning second-placement
correspondence changes both to the other actor's `FirstStone`.  A win
terminates rather than creating a cadence mismatch.  Induction from the
S15 checkpoint proves the claim. ∎

**GAP-OPENING-ALIGNMENT, cadence/history-domain component [PROVEN].**
This component is discharged: S15 accounts for the compulsory real F stone and both first-turn
real S stones in one legal, strategy-consistent role-swapped shadow prefix.
S15.1 proves that, if legality/collision/terminal transfer continues, the
count and phase alignment persists without another opening patch.  This is
the standalone ranked-(c) result.  It does **not** assert that the future
placement transfer succeeds.

## 20. Ranked outcome (b): static two-proxy exact copying is impossible

### 20.1 The class

**Definition 20.1 (`C_static^2`).**  Fix a legal synchronized prefix of the
form constructed in §19, except that either member of the first shadow
`F̂` pair may be designated as the proxy.  Thus:

- `p_S=(0,0)` is the sole shadow-`Ŝ` proxy;
- `q_1,q_2` are `σ`'s legal first `F̂` pair;
- exactly one of `q_1,q_2` is the sole shadow-`F̂` proxy `p_F`;
- the other is `r=T(x)`, the image of the real compulsory opener; and
- `T=t+g`, `g∈D6`, is a bijective isometry taking all remaining real
  stones to the nonproxy shadow stones.

A continuation belongs to `C_static^2` when it promises all of the following:

1. `T,p_S,p_F` remain fixed;
2. at every covered live prefix,
   `X_Ŝ=T[X_S]∪{p_S}` and
   `X_F̂=T[X_F]∪{p_F}`;
3. the real and shadow phases agree;
4. each real S placement `y` is immediately appended as
   shadow `Ŝ@T(y)`;
5. each sequential shadow `F̂@z` prescribed by `σ` is immediately
   realized as real `F@T^{-1}(z)`; and
6. no proxy is retired, moved, backed by a real stone, rebound to a later real
   stone, or replaced by a filler/queue/recode convention.

The promise is against every legal real S continuation from the synchronized
prefix.  It is an exact-copy class, not a definition of every possible
bounded-proxy coupling.

### 20.2 The unavoidable live proxy preimage

**Theorem S16 (static two-proxy collision obstruction) [PROVEN].**  For every
strategy `σ` and every legal `C_static^2` synchronization, the continuation
fails before it can copy the next real S turn.  More exactly, either one of
`σ`'s next two shadow replies has no legal real inverse, or after both
inverses are placed the real S has a legal nonwinning first placement whose
shadow image is an occupied proxy.

*Proof.*  Recall that `r=T(x)` is the nonproxy member of the first shadow
`F̂` pair.

- If `r=q_1`, then `q_1`, as the pair's first coordinate, was legal from
  the only earlier shadow stone `p_S`.  Hence `d(r,p_S)≤8`.
- If `r=q_2`, then at its placement the earlier shadow occupancy was
  `{p_S,q_1}`; here `q_1=p_F`.  Thus
  `d(r,p)≤8` for at least one `p∈{p_S,p_F}`.

In either case select such a proxy `p` and put

`c=T^{-1}(p)`.                                            (20.1)

The proxies are disjoint from all represented shadow stones, so `c` is
empty on the real board.  Isometry and `T(x)=r` give

`d(c,x)=d(p,r)≤8`.                                        (20.2)

Therefore `c∈L(X_F∪X_S)` already at the synchronized real F
`FirstStone` checkpoint.

Now query `σ` after the synchronized shadow prefix from §19.  Shadow `F̂` initially
has two stones.  Its prescribed first reply cannot win with a third stone, so
let the full legal pair be `z_1,z_2`.  Attempt to place
`T^{-1}(z_1),T^{-1}(z_2)` sequentially on the real F turn.

If either inverse is illegal at its required phase, the promised coupling has
already failed.  Otherwise both are placed.  Neither inverse equals `c`,
because `z_1,z_2` are legal shadow placements and hence cannot equal the
already occupied proxy `p=T(c)`.  Real F now has only three stones and real
S has two; shadow `F̂` has only four and shadow `Ŝ` three.  Consequently
neither history is terminal, and the real game passes to S at
`FirstStone`.

The coordinate `c` is still empty and still supported by the permanent
real stone `x`.  Real S may therefore play `c`.  It then owns only three
stones, so the placement is nonwinning.  But item 4 of Definition 20.1 would
have to append

`Ŝ@T(c)=Ŝ@p`,

which is illegal because `p` is already proxy-occupied.  Thus exact
immediate copying fails.

For completeness, this real prefix can always be extended through S's second
placement.  Immediately after `S@c` there are only five other occupied
cells, while `c` has six adjacent cells; at least one is empty, legal at
distance one, and cannot give S six when S then owns only four stones.  The
six radius-one neighbors follow directly from the axial distance enumerator
at `packages/hexo_engine/rust/src/coord.rs:76-95`.

The construction is also inside the literal safe carrier.  S15 bounds the
initial shadow coordinates by hex norm 32.  The next shadow pair has norms at
most 40 and 48, its real inverses have norm at most 64, (20.2) gives
`||c||_h≤8`, and the optional adjacent second coordinate has norm at most
9.  Even the radius-eight legal-update halo has norm at most 72, far inside
`i16`. ∎

S16 does not depend on a preselected witness history that a
strategy-generated invariant might avoid.  Once a `C_static^2` coupling
itself reaches its S15 checkpoint and realizes its actual next `σ` pair,
the adversarial real S computes and plays the live coordinate (20.1).
This strengthening is **[PROVEN]** under Definition 20.1's fixed,
public, pure coupling.

**Corollary S16.1 (global window-exact encoders do not rescue the static
class) [PROVEN].**  Replacing Definition 20.1's explicit isometry by a global
window-exact injection does not avoid S16.

*Proof.*  Round-2 S8.1 proves that every such injection is bijective and has
the form translation composed with D6.  It therefore has the inverse and
distance preservation used in (20.1)–(20.2). ∎

**Exact scope of ranked outcome (b) [PROVEN].**  S16 excludes the minimal
genuine two-proxy repair when the proxies and one bijective isometry are
static and every move is copied immediately one-for-one.  It does not exclude
a coupling that retires or backs `p` before S acts, treats a same-owner
proxy collision as an already-represented move, uses a filler or queue, changes
`T`, employs a non-surjective finite encoding, or stops with a justified
terminal verdict.

## 21. Exact frontier and collision ledger

### 21.1 Two-sided failure sets

At this point it is useful to keep the real surplus and shadow invention
separate.  Put `N_8(∅)=∅`.

**Theorem S17 (two-sided surplus/proxy failure formula) [PROVEN].**  Let
`A,E,P` be finite and pairwise disjoint, with `A` nonempty.  At a
normalized checkpoint let the real occupancy be `A∪E` and the shadow
occupancy be `A∪P`.  For exact identity copying in the normalized
coordinates, define

`Fail_{R→H}={y∈L(A∪E): y∉L(A∪P)}`

and

`Fail_{H→R}={z∈L(A∪P): z∉L(A∪E)}`.

Then

`Fail_{R→H}=C_{R→H} ∪ U_{R→H}`,                           (21.1)

where the disjoint terms are

`C_{R→H}=P∩L(A∪E)`,                                      (21.2)

`U_{R→H}=(N_8(E)\N_8(A∪P))\(A∪E∪P)`,                   (21.3)

and symmetrically

`Fail_{H→R}=C_{H→R} ∪ U_{H→R}`,                           (21.4)

`C_{H→R}=E∩L(A∪P)`,                                      (21.5)

`U_{H→R}=(N_8(P)\N_8(A∪E))\(A∪E∪P)`.                   (21.6)

The `C` terms are occupied-coordinate collisions.  The `U` terms are
empty coordinates whose only support is on the source-only side.

*Proof.*  Take `y∈L(A∪E)`.  It is empty in `A∪E`.  If `y∈P`, its
shadow copy fails exactly by occupancy, giving (21.2).  Otherwise it is empty
in `A∪P`, so shadow legality fails exactly when
`y∉N_8(A∪P)`.  Real legality and that nonmembership imply that its real
support lies in `E`, giving (21.3); the displayed subtraction records
emptiness on both boards.  These cases are disjoint and exhaustive.
Interchanging `E` and `P` proves (21.4)–(21.6). ∎

**Round-2 specializations [PROVEN].**

- With `P=∅`, (21.1) is exactly
  `Γ(A,E)=(N_8(E)\N_8(A))\(A∪E)`.  Equation (21.5) is the separate
  real-surplus collision excluded by S5's “still empty in the real board”
  premise.
- With `E=∅`,
  `Fail_{R→H}=P∩L(A)`: every real move has mapped shadow support, and only
  a proxy collision can stop its immediate copy.  Conversely,
  `Fail_{H→R}=Γ(A,P)`: a shadow strategy reply can be legal solely because
  an invented proxy enlarged the shadow frontier.

Thus proxies can cover part of the real-surplus unsupported set by adding
`N_8(P)`, and real surplus can cover part of the proxy-supported reply set
by adding `N_8(E)`, but each also creates the opposite collision set.  This
trade is an exact one-coordinate rule statement **[PROVEN]**, not a theorem
that either side has a winning strategy.

**Sequential-pair update [PROVEN].**  After a first placement is transferred
successfully to both boards as a represented stone, add it to `A` and
recompute (21.1)–(21.6) before treating the second placement.  If it is
introduced on only one side, update `E` or `P` instead.  If it wins, there
is no second placement.  This is forced by the within-turn update and
win-before-phase-transition rules at
`packages/hexo_engine/rust/src/state.rs:302-337`.

Equations (21.1)–(21.6) diagnose exact copying only.  An own-color collision
might be handled by declaring the move already represented and repairing the
phase/count offset; an opposite-color collision might require recoloring or
recoding.  No such repair is proved here, so those extensions are **[OPEN]**.

### 21.2 S16 is the collision specialization

At the S16 synchronization, normalize by `T^{-1}` or, equivalently, work
on the shadow board with

`A=T[X_F∪X_S]`, `E=∅`, and `P={p_S,p_F}`.

The proxy `p` selected in S16 is empty on the mapped-real board and lies
within eight of `r=T(x)∈A`.  Hence

`p∈P∩L(A)=Fail_{R→H}`.

Its real coordinate is exactly `c=T^{-1}(p)`.  S16 proves more than
nonemptiness of the collision set: the cadence forces real F to move first,
but legal shadow replies cannot consume `p`, so this collision remains live
for S's next turn.  This interpretation is **[PROVEN]**.

### 21.3 A proxy-supported shadow reply really can leave the real frontier

**Lemma S18 (explicit proxy-frontier reply failure) [PROVEN].**  Use

`x=(0,0)`, `a=(0,1)`, `b=(0,2)`,

`p_S=(0,0)`, `p_F=q_1=(1,0)`, `q_2=(2,0)`,

and `T(c)=c+(2,0)`.  The real prefix

`F@x ; S@a,S@b`

and shadow prefix

`Ŝ@p_S ; F̂@q_1,F̂@q_2 ; Ŝ@T(a),Ŝ@T(b)`

are legal and nonterminal.  At the synchronized checkpoint put

`A={(2,0),(2,1),(2,2)}`, `P={(0,0),(1,0)}`.

Then the shadow coordinate `z=(-8,0)` belongs to `Γ(A,P)`, while its real
inverse `T^{-1}(z)=(-10,0)` is illegal.

*Proof.*  Every displayed opening-prefix step after the forced origin is at
distance one from an earlier stone, and no owner has more than three stones.
For the reply coordinate,

`d(z,p_S)=8`,

whereas its distances to the three members of `A` are `10,11,12`.
Thus it is empty and legal solely through the invented opener proxy.  After
translation back, `(-10,0)` has distances `10,11,12` from the entire real
occupancy `{(0,0),(0,1),(0,2)}`, so it is illegal.  If desired, the shadow
pair can be completed at `(-9,0)`, distance one from `z`; four
shadow-`F̂` stones cannot yet win.  All coordinates have magnitude at most
10, and the radius-eight legal-update halo at most 18. ∎

S18 disproves an unconditional “every legal proxy-shadow reply lifts”
lemma.  It does not prove that an alleged winning `σ` must choose this
reply; strategy-specific avoidance of every updated set `Γ(A,P)` remains
**[OPEN]**.

## 22. Win-transfer direction and proxy fabrication

### 22.1 What an isometric live coupling transfers

**Lemma S19 (terminal-direction ledger) [PROVEN].**  Consider an
owner-faithful coupling whose representation invariant holds at every live
checkpoint and through the successfully transferred placement that may create
a terminal checkpoint.  Suppose every real stone is represented under one
translation/D6 isometry `T`, and the only additional shadow stones are
proxies.  Then:

1. a real owner's six maps to a six of the corresponding shadow role, unless
   the shadow game has already terminated;
2. a shadow six containing no proxy maps back to a real six of the
   corresponding owner;
3. a legal shadow-`Ŝ` win, proxy-assisted or not, contradicts the premise
   that `σ` is a winning strategy for shadow `F̂`; but
4. a shadow-`F̂` six containing an invented `F̂` proxy need not give real
   F a six.

*Proof.*  A translation/D6 isometry maps each engine six-window bijectively
to an engine six-window.  Owner fidelity maps every represented stone to the
corresponding owner, proving items 1 and 2 by set inclusion.  Proxies cannot
remove a represented stone or block an already represented window; a
proxy/image collision would instead have broken the live-coupling hypothesis
earlier.

For item 3, the shadow prefix is a legal play against `σ` and all
shadow-`F̂` moves agree with `σ`.  A shadow-`Ŝ` terminal prefix is
therefore a direct counterplay to the alleged winning strategy, regardless of
whether its final six uses `p_S`.  That finite legal counterplay extends to
a total pure shadow-opener strategy by prescribing arbitrary legal moves at
all other decision nodes, so it closes the quantifier in the definition of a
winning `σ`.  For item 4, the inverse window can be
missing the proxy's unoccupied real preimage.  Lemma S20 gives an exact legal
instance. ∎

The qualifier “unless the shadow game has already terminated” in item 1 is
load-bearing.  If an invented `F̂` stone fabricates an earlier shadow win,
the engine exposes no further legal shadow moves
(`packages/hexo_engine/rust/src/state.rs:203-252`) and rejects every attempted
further placement (`packages/hexo_engine/rust/src/rules.rs:11-14`), so the coupling cannot
continue until a later real S six and appeal retroactively to window
monotonicity.

### 22.2 Exact second-placement fabrication gadget

**Lemma S20 (an `F̂` proxy can fabricate the terminal verdict) [PROVEN].**
Let

`p_S=(0,0)`, `p_F=(1,0)`, and `T(c)=c+(2,0)`.

The following real history is legal and nonterminal through every displayed
placement:

`F@(0,0);`

`S@(0,1),S@(0,2);`

`F@(1,0),F@(2,0);`

`S@(0,3),S@(0,4);`

`F@(3,0),F@(4,0)`.                                      (22.1)

Its genuine-proxy shadow is

`Ŝ@(0,0);`

`F̂@(1,0),F̂@(2,0);`

`Ŝ@(2,1),Ŝ@(2,2);`

`F̂@(3,0),F̂@(4,0);`

`Ŝ@(2,3),Ŝ@(2,4);`

`F̂@(5,0),F̂@(6,0)`.                                    (22.2)

The final placement in (22.2), which is the second placement of its turn,
completes a shadow-`F̂` six.  The corresponding final real placement in
(22.1) does not complete a real F six.

*Proof.*  In both histories every normal placement is fresh and at distance
one from an earlier occupied coordinate; within each pair, the second may use
the first.  This is legal under the radius-eight rule.  Before the last shadow
placement, shadow `F̂` owns only
`{(1,0),(2,0),(3,0),(4,0),(5,0)}`; the last placement adds `(6,0)` and
completes exactly that q-axis window.  Shadow `Ŝ` has only five total stones.

At the corresponding real endpoint, F owns only
`{(0,0),(1,0),(2,0),(3,0),(4,0)}`, five stones, and S owns four.  Thus the
real state is nonterminal.  No earlier prefix can contain a six because the
relevant owner then has fewer than six total stones.  The largest coordinate
magnitude is six and the radius-eight legal-update halo at most 14, safely
inside `i16`. ∎

The shadow history (22.2) can be the play of some legal pure second-player
strategy, but S20 does not claim that an alleged *winning* `σ` must select
these replies.  Its conclusion is the narrower rule-level one:

`shadow F̂ win  ⇒  real F win`

is false without a proxy-free-window invariant or a separate repair.
In real coordinates the missing preimage of `p_F` is `(-1,0)`; just before
the final real F turn, obtaining the transferred six would require that
missing cell in addition to both prescribed placements, three cells for a
two-stone turn.  This explains why second-placement terminal timing is a
distinct obligation rather than an ordinary one-coordinate filler case
**[PROVEN for the displayed gadget]**.

### 22.3 Consequence for a stealing proof

**Global proxy transfer [OPEN].**  S15 starts a legal shadow and S15.1 removes
cadence as a future obstruction.  A proof of `NL_F` still must maintain,
after every first placement as well as every completed pair:

- avoidance or repair of the real-surplus frontier/collision set
  `Fail_{R→H}`;
- avoidance or repair of the proxy-frontier/collision set
  `Fail_{H→R}`;
- legal nonterminal strategy-domain histories until a justified stop; and
- transfer of every shadow-`F̂` terminal window containing a proxy.

S16 proves that leaving the minimal proxies static with exact immediate copy
cannot meet this conjunction.  S18 and S20 show that the reverse-frontier and
terminal concerns are real rule phenomena, not bookkeeping formalities.
None of these results proves that every dynamic bounded-proxy coupling fails.

## 23. Result ledger, survivor boundary, and resume point

### 23.1 Ranked outcome assessment

| Claim | Status | Exact scope |
|---|---|---|
| Inherited rule contract | **PROVEN** | Production engine in the safe carrier region; `Z²` for the declared infinite-play idealization |
| D2 non-loss bridge | **PROVEN from CITED** | `NL_F ⇔` S has no winning strategy in the unbounded macro-game |
| Genuine-proxy strategy domain | **PROVEN** | S15 prefix only: the proxy opening is real-legal and the first `F̂` pair is exactly `σ`'s |
| S15 opening synchronization | **PROVEN** | Budget `(β_Ŝ,β_F̂)=(1,1)`; every legal real first S pair; one history-dependent D6 orientation |
| S15.1 permanent cadence repair | **PROVEN** | Conditional on future successful transfer, counts and `FirstStone`/`SecondStone` phases never mismatch again |
| `GAP-OPENING-ALIGNMENT` | **PROVEN** | Cadence/legal-prefix component discharged: both first-turn S stones and the real F opening are represented; global transfer not claimed |
| S16 `C_static^2` obstruction | **PROVEN** | One static proxy per role, one fixed bijective isometry, total exact immediate copy, no retirement/backing/filler/recode |
| S16.1 global window-exact variant | **PROVEN** | Round-2 S8.1 reduces the encoder to the S16 isometry case |
| S17 two-sided failure formulas | **PROVEN** | Pairwise-disjoint normalized `A,E,P`; one exact-copy placement, recomputed sequentially |
| S18 proxy-frontier gadget | **PROVEN** | Exact coordinates; legal shadow reply in `Γ(A,P)` with an illegal real inverse |
| S19 terminal-direction ledger | **PROVEN** | Owner-faithful isometric invariant retained through the resulting, possibly terminal, placement |
| S20 proxy-win fabrication gadget | **PROVEN** | Exact legal histories; shadow second-placement `F̂` win with no real F win |
| Ranked outcome (b) | **PROVEN** | For `C_static^2`: a sharpened bounded-invention obstruction, not all proxy couplings |
| Ranked outcome (c) | **PROVEN** | The two-proxy opening repair is legal and strategy-consistent; cadence remains aligned conditional on successful transfer |
| A global proxy coupling / outcome (a) | **OPEN** | Dynamic proxies, recoding, fillers, and strategy-specific `Γ`-avoidance survive |
| `NL_F` | **OPEN** | D2 is available, but no arbitrary S winning strategy is refuted |

There are no **SKETCH** or **CONJECTURE** results in this round.  The positive
theorem is S15/S15.1; the negative theorem is S16 at Definition 20.1's exact
scope.  S18 and S20 are rule-level stress gadgets, not claims that a winning
`σ` selects their displayed moves.

### 23.2 Exact survivor boundary

**Survivor boundary [PROVEN as a consequence of S16].**  Any coupling that
continues from the S15 synchronization must violate at least one
`C_static^2` premise.  In particular it must use at least one of:

1. retire, move, or real-back a proxy before its live preimage can be played;
2. absorb an own-role proxy collision through a phase/count repair;
3. use a filler, queue, or delayed representation;
4. choose a history-dependent new coordinate map or a non-surjective finite
   encoding;
5. abandon immediate one-for-one copying; or
6. stop earlier under a terminal-faithfulness theorem.

This disjunction follows because keeping every listed premise is exactly the
class S16 excludes.  It does not assert that any survivor mechanism succeeds.

Round-2 S12 remains a causal warning **[PROVEN at its inherited scope]**:
choosing an empty future real F first coordinate before an intervening S turn
lets S occupy it.  Therefore a delayed repair cannot merely preannounce the
coordinate with which it plans to back a proxy.  Round-2 S13's
frontier-only move and S14's terminal six also remain mandatory tests for any
scheme that introduces a queue or literal lag.

### 23.3 Named resume point

**`GAP-PROXY-RETIRE-OR-RECODE` [OPEN].**  Starting from S15's exact
synchronized prefix, give a causal update rule that eliminates or harmlessly
absorbs every live collision

`P∩L(A∪E)`

before the corresponding real opponent can play it, while simultaneously:

1. keeping a genuine legal shadow history consistent with `σ`;
2. preserving S15.1's two-placement phase relation;
3. controlling both updated unsupported sets (21.3) and (21.6) after each
   first placement;
4. avoiding S12 preannouncement;
5. handling real-surplus and proxy collisions with the correct owner; and
6. transferring a proxy-assisted shadow-`F̂` win, especially the
   second-placement pattern of S20.

The first forced checkpoint is the one in S16: after `σ`'s next pair is
realized, the chosen proxy preimage `c=T^{-1}(p)` is still legal for S.
A successful repair must change that fact or explain why playing `c` is
harmless without leaving the legal strategy domain.  S18's `(-8,0)` reply
and S20's final `F̂@(6,0)` are the first explicit reverse-frontier and
terminal regression tests.

## 24. Provenance

**Input state.**  Branch `hunt/gap-raw`, HEAD `57fcbda8`.  This authoring
pass created no commit.

**Required documents read first, in order and in full.**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`; and
3. `STRATEGY_STEALING_REVIEW_ROUND2.md`.

Review Finding 6's exact correction was preserved: one invented proxy of each
color defeats S9.1's no-invention cadence count obstruction, but the review
did not itself construct a legal coupling.  S15 supplies that missing legal
opening prefix.  The review's nonterminal qualifier for S9 and totality
qualification for D1 are also retained.

**Rule sources checked.**  The cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,state,tactics,board}.rs`.
All new constructions use only the inherited origin opening, radius-eight
color-blind legality, sequential pair update, immediate six detection, and
terminal no-move facts, with exact source ranges in §§17, 21, and 22.

**Machine work.**  None.  No Cargo command, Lean build, harness run, or
executable proof search was performed.  No `GAP_RAW_*` file or production
source was edited.  The only intended deliverable written by this session is
`STRATEGY_STEALING_ROUND3.md`.
