# Strategy stealing in engine Hexo, round 4: dynamic proxy rebinding and terminal debt

**Worktree:** `hunt/gap-raw` at input HEAD `5023169f`  
**Date:** 2026-07-17  
**Ranked outcomes:** **(b) achieved for `C_react^<=1` [PROVEN]** and
**(c) achieved for the P5R module `C_shield` [PROVEN at its stated
conditional-class scope]**.  
**Global target:** `NL_F` remains **[OPEN]**.

This round does not produce a global dynamic coupling. It proves two narrower
facts about candidate mechanisms starting from round 3's genuine two-proxy
synchronization.

First, proxy retirement or movement performed before the next real coordinate
is known cannot make a total exact isometric representation safe. The physical
shadow history is connected by radius-eight support. Therefore every
nontrivial total-exact partition of its stones into represented stones and
persistent proxies has a support edge
across the partition; the real preimage of the proxy endpoint is empty, legal,
and maps to an occupied shadow coordinate. After round 3's next transferred F
pair, the two placements of S's next turn are both necessarily nonwinning.
Reapplying the cut after the first repair means a candidate either fails
sooner or uses two coordinate-reactive escapes in that one ordered turn. Thus,
within the zero-lag total-exact
owner-faithful family, arbitrary pre-turn rebinding plus at most one reactive
escape in that next S turn is insufficient. This relaxes `C_static^2`'s fixed-map
restrictions at the shared checkpoint, but it does **not** prove that a
survivor needs infinitely many total rebindings.

Second, `P5R REAL-S-TERMINAL-REFLECTION` admits a phase-sensitive invariant.
A real-only S stone creates a terminal debt in every unblocked six-window that
contains it. The phase-exact prevention condition used by the shield is that
the number of empty cells in each such window exceed the number of S placements
remaining before F next acts; P5R could alternatively be discharged by an
actual shadow terminal certificate. At an F turn, a completed nonwinning physical F pair restores this
shield exactly when it hits every one- and two-hole threatening window; a
transversal of size at most two supplies such service unless F wins earlier.
This gives a nontrivial P5R-safe dynamic class. A legal three-axis fork
has transversal number at least three, so the service condition is not automatic; and
the unconstrained geometric hitting number for permanently fencing all
eighteen windows through one surplus stone is six F blockers. Immediate
geometric reconciliation of unsafe surplus therefore
remains the central open step.

No Cargo command, Lean build, harness, executable search, or production-source
edit was used. No `GAP_RAW_*` file was read or changed in this round.

## 25. Statement boundary and binding inherited contract

### 25.1 Target, determinacy bridge, and status discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real second
player. The target remains

`NL_F : exists a pure strategy sigma_F such that, for every sigma_S, S never wins`.

As in rounds 1--3, "never wins" permits either a finite F win or an infinite
history with neither six. The latter is a meta-level draw in the declared
unbounded-board idealization, not an engine `GameOutcome`.

Round-2 Theorem D2 is inherited at its exact scope:

`NL_F  <=>  S has no winning strategy`.                         (D2)

This is **[PROVEN from the CITED Gale--Stewart open-determinacy theorem]** for
the unbounded Hexo macro-game. D2 supplies only the logical bridge. This round
does not refute an arbitrary winning strategy for S, so it does not establish
`NL_F`.

Every named or load-bearing claim below is marked **PROVEN**, **SKETCH**,
**CONJECTURE**, **OPEN**, or **CITED**. Definitions are stipulative. There are
no machine-verified claims.

### 25.2 Rules used, unchanged [PROVEN]

The round-1 formalization is used verbatim. On the axial hex grid put

`d((q,r),(q',r')) = max(|q-q'|, |r-r'|, |(q-q')+(r-r')|)`.

For finite nonempty occupancy `O`, define

`N_8(O)={c : min_{z in O} d(c,z)<=8}` and `L(O)=N_8(O)\O`.

The only opening is `F@(0,0)`. Along nonterminal real play the owner cadence is

`F ; S,S ; F,F ; S,S ; F,F ; ...`.

A normal placement is legal exactly when it is empty and belongs to `L(O)`.
The first nonwinning placement of a pair is inserted and updates the legal
store before the second is checked. Either owner wins immediately upon filling
one length-six window in direction `(1,0)`, `(0,1)`, or `(1,-1)`; a win on the
first placement suppresses the second.

The production source tie-down is:

- radius eight and its incremental halo:
  `packages/hexo_engine/rust/src/legal.rs:17-18,123-145`;
- opening, occupancy, phase, and terminal rejection:
  `packages/hexo_engine/rust/src/rules.rs:11-44`;
- insertion before legal/window updates:
  `packages/hexo_engine/rust/src/board.rs:83-105`;
- opening and the `1`-then-`2:2` transition, with win checked before phase
  advancement: `packages/hexo_engine/rust/src/state.rs:289-337`;
- the six-cell length, three axes, all-six predicate, and eighteen incident
  windows: `packages/hexo_engine/rust/src/tactics.rs:13-17,21-75,205-208,
  451-485`; and
- terminal states exposing no legal move:
  `packages/hexo_engine/rust/src/state.rs:203-252`.

The executable carrier is `i16` (`packages/hexo_engine/rust/src/coord.rs:9-16`).
General statements below concern the same `Z^2` idealization as D2. Every
explicit finite gadget and the first dynamic obstruction checkpoint are also
bounded inside a stated safe `i16` region.

### 25.3 Standing results and round-3 errata [PROVEN at inherited scopes]

The following boundaries are binding.

- Round-1 S3 refutes identity/deletion legality on its `(8,0)` witness only;
  S4 refutes the one-extra-stone opening alignment only.
- Round-2 S9/S11 exclude their synchronous no-invention/fixed-isometry
  classes. They do not cover genuine proxies or dynamic recoding.
- Round-3 S15 gives a genuine, legal, strategy-consistent two-proxy
  synchronization. Its D6 selector is made functional by fixing an
  enumeration and taking the first surviving orientation.
- S15.1 is only the count/phase invariant under an assumed successful
  common-live one-for-one continuation. It permanently absorbs S4 as a count
  issue; it does not supply later transfer.
- S16 excludes `C_static^2` per synchronization: either the next F
  prescription fails to transfer, or there exists a legal S continuation
  colliding with a fixed proxy.
- S17 is a one-coordinate legality ledger. It cannot see a later real six
  through an older real-only S stone.
- `P5R REAL-S-TERMINAL-REFLECTION` is independent and binding: every real S
  six meeting `E_S`, including an opposite-role proxy collision, must yield a
  legal shadow-`Shat` win no later or be forbidden by the invariant.
- The real and shadow engine histories are append-only. "Retire", "move",
  "back", and "rebind" below alter only the representation. Old physical
  stones retain their occupancy, radius-eight support, blocking, and terminal
  effects.

For the final point, a forward legal placement inserts into the owner map and
pushes the occupied list (`packages/hexo_engine/rust/src/board.rs:88-95`) and
appends the placement history (`packages/hexo_engine/rust/src/state.rs:265-273,
304-307`). The engine exposes an undo API at `state.rs:360-369`; per
post-review erratum (Finding 12) this is an analysis facility used from the
MCTS/search context (`state.rs:283-288`), outside legal forward `Placement`
transitions — not an access-enforced "search-only" operation — and it is not
used by any representation update here. New states start with `Player0` in
`Opening` per `state.rs:149-160` (completing the `289-337` citation).

## 26. Dynamic representation semantics

### 26.1 Physical boards versus current bindings

**Definition 26.1 (genuine rebinding).** At a common-live checkpoint let
`O_R=X_F union X_S` be the physical real occupancy and `O_H` the physical
shadow occupancy. A total exact isometric binding consists of

`T(c)=t+g(c)`, with `g in D6`,

and a partition

`O_H = A disjoint-union P`, where `A=T[O_R]`.                 (26.1)

The map is owner-faithful under the role swap: images of real F stones belong
to shadow `Fhat`, and images of real S stones belong to shadow `Shat`.
Members of `P` are the **current proxies**. They include every physical shadow
stone not currently bound to a real stone, whether or not it was called a
proxy when originally placed.

A rebinding replaces `(T,A,P)` by another triple satisfying (26.1). It does
not change either physical occupancy or either history. Thus backing an old
proxy moves it from `P` to `A`, but the count offset forces some other
persistent shadow stone into `P`. A purported "move" similarly changes which
stone is active in `P`; no stone moves on the engine board.

At the synchronized `FirstStone` checkpoints, full owner fidelity and S15.1's
count offset imply exactly one current proxy of each shadow role. The same is
true after a successfully paired single placement on both boards.

### 26.2 Preemptive rebindings and reactive escapes

**Definition 26.2 (committed representation).** A representation is
*committed for the next real placement* when its `(T,A,P)` is a deterministic
function of the observed history before that coordinate is chosen. In a pure
perfect-information coupling, the real opponent may compute this data from
the coupling rule and the public history. Arbitrarily many representation
updates may occur before commitment; only the last committed triple matters
for the next coordinate.

**Definition 26.3 (coordinate-reactive escape).** Suppose the next real S
coordinate `c` is legal but its committed exact copy `T(c)` is occupied. A
coordinate-reactive escape is any response chosen after observing `c` that
deviates from that committed exact append in order to restore a genuine legal
common-live invariant. It includes:

- changing `T` or the represented/proxy partition;
- backing a same-role proxy and appending a different shadow filler;
- recoding an opposite-role collision; or
- any combined representation repair whose choice depends on `c`.

Pure relabeling before `c` is known is preemptive, not reactive. Leaving the
divergence unresolved until after another placement is a lag/queue scheme,
not an escape that restores the invariant at the current single-placement
checkpoint.

**Definition 26.4 (`C_react^<=1`).** Starting at an S15 synchronization
constructed for the same fixed strategy `sigma`, a member of
`C_react^<=1` may perform any finite sequence of preemptive genuine
rebindings and may use arbitrary legal filler/backing behavior inside one
reactive escape. Every update sequence is pure, causal, deterministic, and
ends in a committed triple before the next real S coordinate is selected. A
reactive escape is charged to the one observed coordinate that triggered it
and ends when the invariant is first restored; one charged episode may not
remain open across the next coordinate. The member promises:

1. a legal shadow history consistent with every `Fhat` prescription of the
   fixed strategy `sigma`;
2. common real/shadow phase and nonterminal status until a sound terminal
   stop;
3. a total exact owner-faithful isometric binding restored and committed
   before each next real S single placement;
4. directional immediate transfer: real `S@c` appends
   `Shat@T(c)` unless a separately charged escape handles `c`, while every
   prescribed shadow `Fhat@z` is realized as real `F@T^{-1}(z)` and leaves an
   exact binding restored; and
5. at most one coordinate-reactive escape during the first real S ordered
   pair after `sigma`'s next F pair.

The class is total over every legal real S choice during that pair. Voluntary
truncation is not permitted. Relative to `C_static^2` at the shared S15
checkpoint, it relaxes the fixed-map/fixed-proxy restrictions: proxy
designations and the global isometry may change, and one collision-dependent
repair is allowed.

(Post-review erratum, R-ST4-REV Finding 4 — syntax/semantics distinction.
Two different collections must not be conflated: the *candidate grammar* —
deterministic update algorithms with legal partial executions — is nonempty,
as witnessed by an explicit hand example in which one backing/filler repair
legally executes; the *extensional class* of members fulfilling all five
promises globally is what S23 proves EMPTY. S23 is a theorem about candidate
rules, stated as: every such candidate fails one of its promises. Also, the
comparison to round 3 is exactly at the shared S15 checkpoint with `q_1` as
proxy — Definition 20.1 allowed either member of the first `Fhat` pair as
proxy, so round 4 is not literally a strict superclass of every round-3
synchronization.)

## 27. The proxy-support cut

### 27.1 Genuine shadow histories are support-connected

For finite occupancy `O`, let `G_8(O)` be the graph with vertex set `O` and an
edge between distinct `u,v` when `d(u,v)<=8`.

**Lemma S21 (support connectivity) [PROVEN].** The graph `G_8(O_H)` of every
nonempty genuine legal shadow prefix is connected.

*Proof.* The opening gives one root vertex. Every later normal placement is
empty and within distance eight of at least one stone already present. Add an
edge to such an earlier support. This includes a pair's second placement,
because the first is inserted and updates the legal halo before the second is
validated. Induction over the append-only placement history gives a connected
spanning tree. The legality and update facts are exactly `rules.rs:34-44`,
`legal.rs:123-145`, and `board.rs:83-105` in the inherited source paths. ∎

This is a physical-history statement. Reclassifying old stones cannot remove
an edge, a vertex, or its support effect.

### 27.2 Every committed exact binding exposes a proxy

**Theorem S22 (proxy-cut exposure) [PROVEN].** Let `(T,A,P)` be a committed
total exact isometric binding at a normal common-live checkpoint. If `A` and
`P` are nonempty, there are `a in A` and `p in P` such that, with

`x=T^{-1}(a)` and `c=T^{-1}(p)`,                         (27.1)

the coordinate `c` is empty and legal on the real board, while its committed
shadow copy `T(c)=p` is occupied.

*Proof.* By S21, `G_8(O_H)` is connected. The nontrivial partition
`O_H=A disjoint-union P` therefore has an edge across its cut: choose
`a in A`, `p in P` with `d(a,p)<=8`. Since `A=T[O_R]` and `p` is outside
`A`, bijectivity gives `c=T^{-1}(p) notin O_R`. The point
`x=T^{-1}(a)` is real-occupied. Isometry gives

`d(c,x)=d(p,a)<=8`.

Thus `c in L(O_R)` at either normal phase. Its prescribed shadow coordinate
is the physically occupied proxy `p`, so the engine occupancy check rejects
the exact append (`packages/hexo_engine/rust/src/rules.rs:34-44`). ∎

**Terminal refinement [PROVEN].** Suppose the next actor is real S and the
coordinate `c` in S22 would complete a real S window `W`. If `p` were owned by
shadow `Shat`, the other five real cells of `W` would already map
owner-faithfully to five `Shat` stones, and `p` would be the sixth cell of the
shadow window `T[W]`; the inherited D6/isometry theorem maps engine windows to
engine windows. The supposedly live shadow would already be terminal.
Hence at a live checkpoint a terminal cut coordinate can only hit an
opposite-role `Fhat` proxy. Leaving the map unchanged then blocks the copied
terminal cell and fails P5R; a different actual shadow-win certificate is
required on that same coupled step.

S22 is invariant under any finite amount of **preemptive** retirement, movement, or
rebinding. Such work merely presents another nontrivial partition of the same
connected physical shadow board. It cannot make all current proxy preimages
illegal. This does not exclude a coordinate-reactive repair after `c` is
revealed.

## 28. Ranked outcome (b): one reactive escape in the tested S turn is insufficient

### 28.1 Two cuts occur before either side can win

**Theorem S23 (`C_react^<=1` obstruction) [PROVEN].** No member of
`C_react^<=1` transfers the complete F-turn/S-turn cycle immediately following
an S15 synchronization. More exactly, for every fixed legal strategy `sigma`
and every legal S15 synchronization constructed for that same `sigma`, either
`sigma`'s next shadow F pair cannot
be realized and followed by restoration under Definition 26.4's candidate
rules, or there is a legal S continuation on which the candidate fails at a
collision or uses at least two separately charged coordinate-reactive escapes.

*Proof.* At synchronization the real role counts are

`(|X_F|,|X_S|)=(1,2)`,

and the shadow role counts are

`(|X_Fhat|,|X_Shat|)=(2,3)`.

The genuine shadow is at `Fhat FirstStone`, so `sigma` supplies its next pair
sequentially. Its first and second placements leave `Fhat` with respectively
three and four stones, hence neither can win. Try to realize the two
prescriptions on the real F turn, with any permitted preemptive rebinding. If
either transfer or its required exact restoration fails, the first alternative
of the theorem holds. If both succeed, real F has three stones and also cannot
win. Both histories are live,
at S/`Shat FirstStone`, with counts

`real (F,S)=(3,2)`, and `shadow (Fhat,Shat)=(4,3)`.          (28.1)

Commit whatever total exact binding the scheme chooses. There is one current
proxy of each shadow role, so S22 supplies a legal collision coordinate
`c_1`. Real S owns only two stones before placing it; `S@c_1` is necessarily
nonwinning. Its committed shadow target is occupied, so restoring Definition
26.4's invariant requires a first coordinate-reactive escape. If no such
escape restores the invariant, totality already fails. Otherwise a successful
escape can append exactly one `Shat` stone: any additional four-placement
shadow cycle contains a prescribed `Fhat` pair that cannot be transferred
while the real game remains at S `SecondStone`, and is therefore an earlier
failure of Definition 26.4 items 2 or 4 (post-review erratum, R-ST4-REV
Finding 3 — the one-append fact follows from the immediate-transfer and
real-turn premises, not phase equality alone). Afterwards the counts are

`real (F,S)=(3,3)`, and `shadow (Fhat,Shat)=(4,4)`,          (28.2)

and both histories are still live at `SecondStone`.

Allow the repair to change the isometry, back either proxy, choose a legal
filler, and perform any finite number of further updates before committing for
the second coordinate. Once the exact invariant is restored, (28.2) again
leaves one current proxy of each role. Apply S22 to the final committed
binding. It supplies a real-empty legal coordinate `c_2`. It is automatically
different from the now occupied `c_1`, so it is legal at `SecondStone`.
This gives S only its fourth stone, and the associated single shadow append
could give `Shat` at most its fifth. No terminal stop is possible on either
side. The occupied committed target of `c_2` therefore forces a second
coordinate-reactive escape; if that repair is unavailable, totality fails
instead. Either branch violates Definition 26.4. ∎

The argument counts same-role backing plus a filler as an escape; calling it
"already represented" does not avoid the result because the filler becomes a
new persistent proxy and the restored partition is cut again. A scheme that
does not restore a total exact binding between `c_1` and `c_2` is a lag/queue
scheme outside `C_react^<=1`.

### 28.2 Literal-carrier bound for the forced checkpoint [PROVEN]

The S15 shadow prefix has hex norm at most 32. The next legal `sigma` pair has
norm at most 40 and then 48. Under any exact rebinding, `T(0)` is an existing
represented `Fhat` stone. Consequently a first real inverse has norm at most
`40+16=56`; after a possible inter-placement rebind, the second has norm at
most `48+40=88`. Before S acts, both `T(0)` and every shadow proxy have norm at
most 48, so `||c_1||_h<=96`. A legal shadow filler after `c_1` has norm at
most 56; `Fhat` has not moved, so the second cut gives `||c_2||_h<=104`.
Every radius-eight update halo used in the proof has norm at most 112. These
coordinates are safely inside `i16`.

### 28.3 Why the conclusion is not "unbounded total rebinding"

**Lemma S24 (finite horizon of a fixed winning strategy) [PROVEN].** From a
fixed finite Hexo checkpoint, if a fixed pure shadow strategy `sigma` wins
against every legal counterplay, then there is a finite `N_sigma` such that it
wins within `N_sigma` further single placements on every counterplay.

*Proof.* Fix `sigma` and retain all nonterminal prefixes compatible with it.
The resulting rooted tree is finitely branching: every finite Hexo position
has a finite legal set, as proved in round-2 D1 from the radius-eight rule.
If the tree had nodes at arbitrarily large depths, the elementary finite-
branching path argument would choose successively a child with descendants at
arbitrarily large remaining depth, producing an infinite branch. That branch
would contain no `Fhat` win, contrary to `sigma` being winning. Hence the
nonterminal tree has bounded depth. ∎

Accordingly, a successful placement-granular repair that performs only
finitely many representation operations per placement could still use a
finite, `sigma`-dependent total number before a sound terminal stop. S23 proves
that **among total exact isometric schemes insisting on zero-lag restoration,
one turn-level reactive escape is insufficient: the candidate either fails at
an occupied-target repair or, if it restores after `c_1`, needs a separately
charged escape at `c_2`**. It does not prove that zero lag
itself is necessary, does not exclude every finite total budget
`K>=2`, and it does not exclude lag, a non-isometric finite encoding, or a
window-certificate representation. Those broader statements remain **[OPEN]**.

This is ranked outcome (b) at the exact scope of Definition 26.4. It is a
dynamic obstruction, not an outcome theorem.

## 29. P5R is a terminal-memory obligation, not a legality obligation

### 29.1 A full legal realization of the review witness

The round-3 review gave a local configuration in which an older `E_S` stone
is invisible to S17's one-coordinate failure sets. The following embeds that
configuration in two complete legal prefixes and begins with an exact S15
synchronization.

**Lemma S25 (reachable older-surplus terminal failure) [PROVEN].** Consider
the real history

```text
F@(0,0);
S@(7,0),S@(8,0);
F@(-1,1),F@(-2,1);
S@(2,1),S@(2,2);
F@(-3,1),F@(-4,1);
S@(2,3),S@(2,4);
F@(-5,2),F@(-6,2).
```

and the shadow history

```text
Shat@(0,0);
Fhat@(-1,0),Fhat@(-2,0);
Shat@(5,0),Shat@(6,0);
Fhat@(-3,1),Fhat@(-4,1);
Shat@(2,1),Shat@(2,2);
Fhat@(-5,1),Fhat@(-6,1);
Shat@(2,3),Shat@(2,4);
Fhat@(-7,2),Fhat@(-8,2).
```

Both are legal and nonterminal, and both end with the S role at
`FirstStone`. Their initial real three/shadow five placements are exactly an
S15 prefix with

`q_1=(-1,0)`, `q_2=(-2,0)`, and `T(c)=c+(-2,0)`.

Now couple the next S-role pair as

```text
real:   e=(2,5), then y=(2,6)
shadow: f=(0,1), then y=(2,6).
```

The final real placement wins on `{(2,1),...,(2,6)}`. The shadow remains
nonterminal: it lacks `(2,5)` and has no other `Shat` six.

*Proof.* Every displayed nonopening coordinate is fresh and has hex norm at
most eight, so the permanent origin alone supplies radius-eight legality; the
listed pairs also have the required owner cadence. Before the last pair, the
longest real-S and shadow-`Shat` axis run has length four,
`(2,1),...,(2,4)`. On the real board, F's longest run is the four cells
`(-1,1),...,(-4,1)`; on the shadow board, `Fhat`'s longest is
`(-3,1),...,(-6,1)`. The remaining same-role cells lie on distinct lines or
in runs of length at most two. Hence no earlier prefix is terminal. The shadow
filler `(0,1)` is adjacent to its opener proxy,
and `(2,6)` is legal on both boards from common `(2,4)` at distance two.
The real final cell completes the displayed R-axis window. The shadow has
`(2,1),...,(2,4),(2,6)` with a gap, while its off-line stones at `(0,0)`,
`(0,1)`, `(5,0)`, and `(6,0)` complete no window. The largest update halo
has norm at most 16, safely inside `i16`. ∎

The shadow `Fhat` pairs on this one branch can be extended to a legal pure
strategy by fixing arbitrary legal prescriptions off the branch. That
strategy is **not** claimed winning. Likewise, the later histories are not
claimed to retain one global isometry. S25 is a rule-level reachability
stress test for any dynamic filler/lag/recode scheme, not a counterexample to
every coupling.

### 29.2 Exact next-placement terminal demand and supply

Let `W_6` be the family of engine six-windows. At a common-live S/`Shat`
single-placement checkpoint, let `E_S` be the real-only S stones already
present. Define

`D_R^E = { y in L(O_R) : exists W in W_6,`

`                         W\{y} subseteq X_S and W intersect E_S != empty }`,

and

`D_H = { z in L(O_H) : exists V in W_6, V\{z} subseteq X_Shat }`.   (29.1)

Thus `D_R^E` is the set of immediate real wins whose five older stones include
real-only surplus, while `D_H` is the set of immediate physical shadow wins.

**Theorem S26 (next-placement P5R ledger) [PROVEN].** Suppose a handler assigns
one actual legal shadow placement `u(y)` to every next legal real S placement
`y`, after any representation-only update. Absent an earlier justified
terminal stop, every next real S win meeting the current `E_S` reflects to a
shadow-`Shat` win no later if and only if

`u(y) in D_H for every y in D_R^E`.                         (29.2)

*Proof.* For `y in D_R^E`, the engine declares the real game terminal
immediately after `y`. There is no later real second placement or F service
turn. The phase-aligned shadow can reflect the result no later exactly when
its associated actual append `u(y)` fills a physical `Shat` window, which is
the definition of `u(y) in D_H`. The converse is immediate from the same
per-placement win predicate. Recompute both sets after any nonwinning first
placement, because it physically changes occupancy before `SecondStone`. ∎

The sets are finite at every checkpoint. Each member of `E_S` lies in exactly
eighteen windows, and each such window has at most one empty terminal demand;
the engine's `WINDOWS_PER_PLACEMENT=18` is at
`packages/hexo_engine/rust/src/tactics.rs:13-17`.

**Lemma S26.1 (physicality of reconciliation) [PROVEN].** Changing only a
binding, proxy label, or current coordinate map cannot change `D_H` or an
engine terminal verdict. A P5R-valid reconciliation must establish an actual
shadow window certificate, or a physical real F stone must block the real
window before S fills it.

*Proof.* `D_H` depends only on the physical shadow board, its legal-support
store, and actual `Shat` occupancy; the terminal predicate depends on actual
owner occupancy in a six-window (`rules.rs:34-44` and
`tactics.rs:205-208,451-485`). Representation data is absent from all of those
predicates. ∎

In particular, binding a real-only stone to an arbitrary off-line shadow
filler preserves cardinality but not terminal geometry. S25 is the exact
counterexample.

## 30. A phase-sensitive P5R shielding invariant

### 30.1 Certified common stones and terminal debt

**Definition 30.1 (window-certified representation).** At each common-live
checkpoint, choose a current translation/D6 isometry `T` and a set
`C_S subseteq X_S` such that every `c in C_S` has the actual shadow stone
`Shat@T(c)`. Put

`E_S=X_S\C_S`.

Only this S-role certificate is required by the P5R module; the other coupling
obligations may impose stronger data. If a representation update removes a
cell from `E_S`, it must really establish this physical certificate. Merely
renaming a filler does not suffice by S26.1.

A real window `W` is **E-live** when

`W intersect E_S != empty` and `W intersect X_F = empty`.       (30.1)

Put

`delta(W)=6-|W intersect X_S|`.                              (30.2)

At a nonterminal position every E-live window has `delta(W)>=1`. Let `m` be
the number of S placements remaining before F next acts:

- `m=2` at real S `FirstStone`;
- `m=1` at real S `SecondStone`; and
- `m=0` throughout the intervening F turn.

For a contemplated S placement, write `m^+=m-1` for the scheduled number of
S placements that would remain before F acts. This bookkeeping value is used
for the immediate post-placement admission test even when the placement is
terminal and the engine therefore does not advance its stored phase.

**Definition 30.2 (deadline shield).** The representation is shielded when

`delta(W)>m for every E-live W`.                            (30.3)

The strict inequality is the terminal deadline. A window with
`delta(W)<=m` can in principle be filled before F receives a physical move.

### 30.2 Preservation during an S turn

**Lemma S27 (deadline-shield induction) [PROVEN].** Suppose (30.3) holds
before an S placement. For every already E-live window it continues to hold
after the physical append with the post-placement value `m^+`, whether or not
the append is terminal elsewhere. A representation
update that newly adds a real S stone `e` to `E_S` is admissible exactly when
every newly E-live, F-unblocked window through `e` satisfies (30.3) in the
post-placement bookkeeping state. Shrinking `E_S` by a valid physical
certification is harmless.

*Proof.* An S placement reduces `delta(W)` by one if it lies in `W`, and by
zero otherwise. At the same transition, the number of S placements remaining
before F acts also falls by one. Thus the strict inequality for every old
E-live window is preserved. Adding `e` can make previously irrelevant windows
E-live, and (30.3) is precisely their missing condition. Removing certified
surplus can only remove E-live windows. A physical F blocker is permanent and
also only removes windows from the live family. ∎

The review witness is rejected at the correct instant. After first-placement
`e=(2,5)`, its R-axis window has `delta=1` while S still owns the
`SecondStone`, so `m=1` and `1>1` fails. There is no F "surplus tempo" between
`e` and `y=(2,6)`.

### 30.3 The conditional dynamic class

**Definition 30.3 (`C_shield`).** A P5R shielding module belongs to
`C_shield` when it uses only representation-level rebindings and physical
legal game placements, retains its certificate through the coupled placement
that may create a terminal state, and satisfies these update rules:

1. before and after every live S-role single placement it maintains (30.3);
   the post-placement test is also required for a terminal real placement
   unless that placement is certified common and already reflected in a
   shadow terminal window;
2. a newly real-only S stone is admitted only by S27's phase-sensitive test,
   evaluated immediately after its physical placement even if the real engine
   has just become terminal;
3. a stone leaves `E_S` only after actual window-certified representation;
4. during the following F turn, it either validly reconciles enough surplus
   or physically blocks enough E-live windows to restore `delta(W)>2` before
   the next S `FirstStone`; and
5. every real S placement classified common is actually appended at `T(y)` so
   common-only sixes map to physical shadow windows.

**Theorem S28 (P5R for `C_shield`) [PROVEN].** Every continuation satisfying
Definition 30.3 reflects a real S win to a legal shadow-`Shat` win no later.
In fact, no real S six can meet `E_S`; every real S terminal window is
common-only and maps under `T` to a shadow terminal window on the same coupled
placement.

*Proof.* At the start of each S turn, item 4 gives the deadline shield with
`m=2`. Lemma S27 preserves it through the first placement with `m=1` and
through the second with `m=0`, including every newly admitted surplus cell.
If an E-live window became full, it would have `delta=0`, contradicting
`delta>m>=0`. A real S winning window therefore meets no `E_S`. All six of its
cells lie in `C_S`; item 5 and the window-preserving isometry put six actual
`Shat` stones in `T[W]` no later than the coupled terminal append. ∎

**Nonvacuity witness from an S15 prefix [PROVEN].** The real prefix

```text
F@(0,0); S@(1,1),S@(2,1); F@(3,0),F@(4,0);
S@(0,5),S@(1,5)
```

and shadow prefix

```text
Shat@(0,0); Fhat@(-1,0),Fhat@(-2,0);
Shat@(-1,1),Shat@(0,1); Fhat@(-3,0),Fhat@(-4,0);
Shat@(0,2),Shat@(-1,5)
```

are legal, nonterminal, end with the F role at `FirstStone`, and begin with an
exact S15 synchronization under `T(c)=c+(-2,0)`. Initialize the P5R module
with certified common S cells `{(1,1),(2,1),(1,5)}` and the genuine real-only
cell `E_S={(0,5)}`; shadow `(0,2)` is the off-line filler. Every window
through `(0,5)` contains at most two real S stones, so
`delta>=4>2`; the surplus persists safely into the next S turn without a
permanent cage. This is deliberately a P5R-module witness, not a full
strategy-stealing coupling satisfying P0--P6.

All displayed cells are fresh and have hex norm at most six, so the origin
supports every normal placement. The owner order gives the stated common
phase, and every role has fewer than six stones. Finally,
`T(1,1)=(-1,1)`, `T(2,1)=(0,1)`, and `T(1,5)=(-1,5)`, while
`T(0,5)=(-2,5)` is absent from the shadow. These observations prove the
claimed legality, nonterminality, and certificate.

(Post-review erratum, R-ST4-REV Finding 8 — membership horizon.
`C_shield` membership means a COMPLETE trace satisfying the admission and
service rules until a justified stop, not merely a module state. The witness
above establishes a reachable shielded state; the review supplies the
terminating extension making the class literally nonempty as a trace class —
keep the same `T` and append, from the witness endpoint:
real F `(-1,0),(-2,0)` / shadow `Fhat (3,0),(4,0)`;
real S `(0,1),(3,1)` / `Shat (-2,1),(1,1)`;
real F `(-3,0),(-4,0)` / `Fhat (5,0),(6,0)`;
real S `(4,1),(5,1)` / `Shat (2,1),(3,1)`.
All cells are fresh, origin-supported, and nonwinning until the final second
placement, which completes the common-only real window `r=1, q=0..5` and its
shadow image `r=1, q=-2..3` simultaneously; throughout, every E-live window
through `(0,5)` contains at most two S stones — the only possible axis
companions are `(1,5)`, `(0,1)`, and `(4,1)` — so `delta>=4` holds until the
sound terminal stop.)

This is ranked outcome (c) at an exact, nonvacuous conditional-class scope.
The class permits real-only S stones to persist while their windows are far
from completion; it does not require a six-stone permanent cage around every
surplus cell. What remains open is whether the full proxy coupling can always
satisfy the admission and F-service rules while also obeying `sigma` and P3.

## 31. Exact cost of physical F service

### 31.1 The urgent-window transversal

At an F `FirstStone` checkpoint, let

`U_E={W : W is E-live and delta(W)<=2}`.                    (31.1)

For `W in U_E`, its **hole set** is

`H_W=W\X_S`.

Because `W` is F-unblocked, every member of `H_W` is physically empty, and
`|H_W|` is one or two. Define the transversal number

`tau_E = min{|K| : K intersects H_W for every W in U_E}`.  (31.2)

The family is finite, so the minimum exists; put `tau_E=0` when `U_E` is
empty.

**Theorem S29 (two-stone service criterion) [PROVEN].** Hold the semantic sets
`C_S,E_S` fixed throughout the F turn: no cell is reclassified, admitted, or
reconciled. A completed nonwinning F pair restores the
deadline shield for the next S `FirstStone` exactly when its two coordinates
contain a transversal of the urgent hole family. Consequently, if
`tau_E<=2`, F can either win earlier during the attempted service or complete
a restoring pair; if `tau_E>2`, every nonwinning F pair leaves the shield
unrestored, and S can either win earlier elsewhere or complete an E-meeting
window on its next turn.

*Proof, sufficiency.* Choose a minimum transversal `K` of size at most two.
Each selected hole of a four- or five-stone S window lies at line distance at
most two from an existing S stone, hence is a legal F placement under the
radius-eight, color-blind rule. Place the distinct members of `K`; if fewer
than two are required, use a fixed enumeration to choose any fresh legal
padding coordinate. If either placement wins for F, service ends soundly.
Otherwise the ordered pair completes. Every urgent window receives a
permanent F blocker. Every E-live window left unblocked had `delta>=3` before
the pair and still does, so `delta>2` holds when S next reaches `FirstStone`.

*Proof, necessity.* A completed pair restores the shield only if every urgent
window contains one of its F stones. Such a stone must belong to `H_W`, since
the other cells of `W` are S-owned. Thus the pair's coordinates must be a
transversal. If `tau_E>2`, any nonwinning F pair misses the hole set of some
`W in U_E`. At the next S turn, the one or two holes of this unblocked window
are legal: with four or five S stones already in a length-six line, each hole
is within distance at most two of S support. S fills them in order. With one
hole it wins on the first placement; with two, the first is nonwinning unless
it wins elsewhere already, and the second completes `W`. ∎

This is an exact real-board service theorem, not yet a coupling update. The
two coordinates may conflict with the real inverses of `sigma`'s shadow
prescriptions. Proving that the P3 transfer can be recoded to use the
transversal, or that reconciliation reduces `tau_E` first, remains **[OPEN]**.

### 31.2 A legal three-axis fork defeats blanket two-tempo fencing

**Lemma S30 (three-axis labeled real-board surplus fork) [PROVEN].** The
following real history is legal and nonterminal:

```text
F@(0,0);
S@(1,1),S@(2,1);       F@(-7,0),F@(-6,2);
S@(3,1),S@(0,2);       F@(-5,4),F@(-4,6);
S@(0,3),S@(0,4);       F@(-2,7),F@(7,0);
S@(1,0),S@(2,-1);      F@(6,-2),F@(5,-5);
S@(3,-2),S@(-4,4);     F@(4,-6),F@(2,-7);
S@e=(0,1),S@(-4,5).
```

At the resulting F `FirstStone` checkpoint, impose the abstract P5R label
`E_S={e}` and label every other S stone common. The
three E-live deficit-two windows

```text
W_Q  = {(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)},
W_R  = {(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)},
W_QR = {(0,1),(1,0),(2,-1),(3,-2),(4,-3),(5,-4)}
```

have pairwise-disjoint hole pairs. Hence `tau_E>=3`. No physical F pair can
block all three, and S wins through an unblocked member on its next turn.

*Proof.* Every post-opening coordinate is fresh and has hex norm at most
seven. The origin therefore supports every listed placement directly, so the
radius-eight rule and the displayed `1`-then-`2:2` cadence hold. Along each of
the three S axes through `e`, S owns offsets `0,1,2,3`; the displayed positive
length-six window has precisely its offsets `4,5` empty. No old F stone lies
in any of the three windows. Their holes are

`{(4,1),(5,1)}`, `{(0,5),(0,6)}`, and `{(4,-3),(5,-4)}`,

which are disjoint.

There is no earlier S six. On `r=1` and `q=0`, the maximum consecutive runs
have length four. On `q+r=1`, the q-coordinates are `-4,0,1,2,3`, whose
maximum consecutive run is again four; every other S-axis line contains at
most two stones. For F, all q-coordinates are
distinct; all q+r values are distinct except that `(0,0)` and `(5,-5)` share
one axis line; and on `r=0` the three stones have q-coordinates `-7,0,7`.
Thus every old F six-window contains at most two F stones. Any next F pair can
raise that count to at most four and cannot terminate the game.

Because two placements hit at most two of the three disjoint hole pairs, one
displayed window remains unblocked. Its offset-4 hole is adjacent to S's
offset-3 stone. That first append is nonterminal: on `W_Q`, `(4,1)` only
extends the selected run to five and its q/QR cross-lines contain no winning
run; on `W_R`, `(0,5)` extends the selected run to five while its r-line
contains only the remote `(-4,5)` and its QR cross-line has no old S stone;
on `W_QR`, `(4,-3)` extends that selected run to five and its q/r cross-lines
contain no old S stone. The offset-5 hole is then adjacent to the first. S
legally fills it and wins on the selected E-meeting window. All possible
winning holes have norm at most six and their update halos at most fourteen;
the whole displayed gadget, including its norm-seven setup cells, has halo at
most fifteen. Everything is safely inside `i16`. ∎

S30 does not claim that an arbitrary coupling must reach this semantic
`E_S` designation or that a genuine shadow certificate for this entire prefix
has been supplied. It proves that the proposed **abstract labeled real-board
service rule** is not universally maintainable over all legal labeled
checkpoints. A survivor must reconcile/recode `e`, obtain a justified earlier
terminal stop,
or use a terminal-reflecting shadow response; the phrase "F has two surplus
tempos" is not a fencing proof.

### 31.3 The permanent-fence geometric hitting number is six

Call a real-only S stone `e` **permanently F-fenced** when every one of the
eighteen engine windows through `e` contains a real F stone.

**Lemma S31 (permanent-fence hitting number) [PROVEN].** Every physical
permanent F fence of all windows through one S-owned cell contains at least six
F stones. On an otherwise available neighborhood, six suffice; equivalently,
the unconstrained geometric hitting number is exactly six.

*Proof.* Fix one of the three axis directions `v`. The two extreme windows

`{e-5v,...,e}` and `{e,...,e+5v}`

intersect only at `e`, which is S-owned. Blocking both therefore requires at
least one F stone on the negative side and one on the positive side of that
axis. The three axis lines meet only at `e`, so their off-center blockers are
distinct: at least six are necessary. The six adjacent cells

`e-v, e+v` for the three axes

suffice as a geometric hitting set, because every length-six interval through
`e` contains the adjacent cell on at least one side. If those six cells are
empty, each is individually legal while it remains empty because `e` is
adjacent; if F eventually occupies all six before S takes one, they produce
the physical fence. This does not prove that F can install them across three
interrupted F turns. ∎

Once installed, such a fence persists by append-only occupancy and makes the
stone irrelevant to P5R forever. It is a valid local way to remove that
stone's P5R debt, but by itself does not establish every `C_shield` update rule.
S31 proves that a fence built from scratch cannot be installed during one
two-stone F turn. Existing blockers can reduce the additional cost, while
S-occupied candidate blocker cells can make a particular six-cell construction
unavailable.

## 32. Dynamic design-space map

| Candidate mechanism | Status | Exact disposition |
|---|---|---|
| Arbitrary pre-turn retirement/movement as the only collision repair | **PROVEN insufficient for total immediate exact copying** | S22: every final committed proxy/common partition still has a radius-eight cut edge; later reactive repair remains possible |
| At most one coordinate-reactive escape in the next S pair within `C_react^<=1` | **PROVEN impossible** | S23: in the zero-lag total-exact class, the nonwinning first repair restores a partition that is cut again at `SecondStone` |
| One zero-lag reactive repair after every single placement | **OPEN** | Must prove a new genuine binding/filler exists after each observed coordinate and preserves all terminal ledgers |
| A fixed finite total rebind budget `K>=2` | **OPEN** | S24 blocks an inference to unbounded total work; a fixed winning `sigma` has a finite uniform horizon |
| Leave a first-placement S divergence unresolved through `SecondStone` | **OPEN subject to P5R** | S25/S27 reject it whenever an E-live window has `delta<=m`; the review witness has `delta=m=1`. `C_shield` is one sufficient module; S26 same-step terminal supply is the other stated route (post-review erratum, Finding 12) |
| Bind the missed real stone to an arbitrary same-owner shadow filler | **PROVEN insufficient for P5R** | S26.1: labels do not create an actual shadow six |
| Fence urgent E-windows with F's next pair | **PROVEN via `tau_E<=2` as a real-board service** | S29: a nonwinning pair must be a transversal; an earlier F win is a sound stop; P3 compatibility is open |
| Assert two F placements always suffice on every labeled real-board service state | **PROVEN false** | S30 supplies three disjoint urgent hole pairs; no full shadow certificate is claimed for that fork |
| Permanently fence each real-only S stone | **PROVEN P5R-safe; geometric minimum six total blockers** | S31; cell availability, installation, and P3 compatibility are open |
| Non-isometric/window-certificate recoding | **OPEN** | It must be physical and simultaneous-threat faithful; a global window-exact injection collapses back to an isometry by round-2 S8.1 |
| Physically erase, recolor, or move an old proxy | **Not a legal mechanism** | Legal histories are append-only; representation retirement cannot change engine occupancy |

The map isolates two different meanings of "dynamic". Within a committed
total-exact binding, preemptive changes alone cannot eliminate S22's cut
exposure because S chooses after commitment. Reactive changes can escape one
collision, but an S23 survivor must use a separate repair per placement, an
explicit lag, or a non-total/non-isometric encoding. P5R then constrains what
that restoration may call a reconciliation: actual window geometry, not only
equal counts, must be repaired.

## 33. Result and obligation ledgers

### 33.1 New result ledger

| Claim | Status | Exact scope |
|---|---|---|
| S21 support connectivity | **PROVEN** | Every NONEMPTY genuine legal shadow prefix; graph edges have distance at most eight |
| S22 proxy-cut exposure | **PROVEN** | Any committed total owner-faithful translation/D6 binding with nonempty represented and proxy parts |
| S22 terminal refinement | **PROVEN** | A real-S terminal cut cannot hit a same-role proxy at a common-live checkpoint; opposite-role hit needs P5R repair |
| S23 `C_react^<=1` obstruction | **PROVEN** | First S pair after the next successfully transferred `sigma` pair; arbitrary preemptive rebinding and one broad reactive escape allowed |
| Literal-carrier bound | **PROVEN** | The S23 forced checkpoint and halos have hex norm at most 112 |
| S24 finite winning-strategy horizon | **PROVEN** | Fixed winning `sigma`, fixed finite checkpoint, finitely branching Hexo tree |
| S25 embedded P5R witness | **PROVEN** | Two exact legal histories, S15 initial synchronization, later dynamic divergence; the extending strategy is legal but not claimed winning |
| S26 next-placement P5R ledger | **PROVEN** | Immediate real wins through already present `E_S`, one legal shadow append, common-live checkpoint |
| S26.1 physicality | **PROVEN** | Representation-only changes cannot alter physical terminal supply |
| S27 deadline-shield induction | **PROVEN** | Phase-sensitive E-live deficits during one real S turn |
| S28 P5R for `C_shield` | **PROVEN** | Conditional dynamic class satisfying certified admission and service rules |
| S29 two-stone service criterion | **PROVEN** | Pure physical F fencing with `C_S,E_S` FIXED during the F turn and no reconciliation; nonwinning pairs are exact transversals, with an earlier F win a sound alternative |
| S30 three-axis fork | **PROVEN** | Abstract labeled real-board service state only; displayed legal real history, three disjoint deficit-two hole sets, no full shadow certificate claimed |
| S31 permanent-fence hitting number | **PROVEN** | All eighteen windows through one S-owned surplus cell; lower bound six, attained on an available neighborhood |
| Ranked outcome (b) | **PROVEN** | Dynamic class `C_react^<=1`; not all bounded-total or lagged schemes |
| Ranked outcome (c) | **PROVEN** | P5R alone for nontrivial conditional class `C_shield`; universal maintenance not claimed |
| Global dynamic coupling / outcome (a) | **OPEN** | No update rule simultaneously discharges P0--P6 and P5R for every continuation |
| `NL_F` | **OPEN** | D2 is available, but no arbitrary S winning strategy is refuted |

There are no **SKETCH** or **CONJECTURE** results in this round. "Conditional
class" in S28 means that the update rules define the class and prove P5R for
every member; it does not assert that a member satisfying the other coupling
obligations exists for every `sigma`.

### 33.2 Full obligation ledger after round 4

| Obligation | Status | Round-4 disposition |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **PROVEN at S15 prefix; OPEN globally** | Every later filler/recode must remain one genuine legal `sigma`-consistent shadow history |
| `P1 OPENING/CADENCE` | **PROVEN for cadence/legal-prefix component** | S15/S15.1 remain binding; only assumed common-live transfer preserves later phase equality |
| `P2 REAL->SHADOW` | **OPEN** | S22 proves every committed total owner-faithful translation/D6 isometry, at a normal common-live checkpoint with nonempty represented and proxy parts, has an adversarial occupied target; non-total, nonisometric, lagged, and window-certificate maps remain open (scope per Finding 12) |
| `P3 SHADOW->REAL` | **OPEN** | Reverse frontier, collision, and compatibility with S29's service cells remain unsolved |
| `P4 COLLISION` | **OPEN globally** | In the zero-lag total-exact family, preemptive-only management and at most one repair in the tested S pair are excluded; per-placement recoding survives |
| `P5 SHADOW-F-TERMINAL` | **OPEN** | Round-3 S20's proxy-fabricated second-placement win still requires transfer or prevention |
| `P5R REAL-S-TERMINAL-REFLECTION` | **PROVEN in `C_shield`; OPEN globally** | Exact deadline and service conditions are known; unsafe surplus still needs immediate geometric reconciliation |
| `P6 CAUSALITY` | **OPEN globally** | S22 exploits every precommitted total exact map (same S22 premises as the P2 row); repairs must wait for the coordinate without exposing a future F placement as in S12 |

The ledger also retains round-3 S18's proxy-supported shadow reply and
round-2 S13/S14's frontier/lag regressions. Neither the shielding module nor
the cut theorem discharges them.

## 34. Named resume point and provenance

### 34.1 `GAP-ZERO-LAG-WINDOW-RECODE / P5R-SERVICE` [OPEN]

(Rewritten per post-review erratum, R-ST4-REV Finding 11, MAJOR: the
original six-item "do one of the following" list mixed three alternative
branches with three mandatory obligations under a single disjunction, and
overstated when checkpoint (28.1) is forced. The corrected quantifier
structure is:)

For every alleged-winning `sigma`, every relevant S15 synchronization or
later strategy-generated live prefix, and every legal real-S continuation,
the candidate must select one valid branch —

- **(A)** zero-lag repair for the current single placement: after observing
  the collision coordinate, append one legal `Shat` stone and restore a
  physical, owner-faithful representation before `SecondStone`;
- **(B)** an explicit lag/queue whose unresolved `E_S` satisfies `delta>m`
  at every intermediate phase; or
- **(C)** a same-step actual physical shadow terminal certificate
  satisfying S26, not an off-line rebinding —

**and, in all branches, discharge all of the following obligations:**

4. before the next S turn, prove that valid reconciliation reduces every
   residual threat or that the P3-compatible real F pair realizes a
   transversal with `tau_E<=2`;
5. retain every old proxy/filler in both frontier and terminal calculations;
   and
6. simultaneously pass S18's reverse-frontier reply, S20's proxy-assisted
   `Fhat` second-placement win, S12's preannouncement collision, and the
   embedded S25 older-surplus terminal test.

Before this fork, the candidate must either transfer `sigma`'s next F pair
with a genuine restored invariant or give the corresponding earlier
P3/terminal repair: checkpoint (28.1) is CONDITIONAL on success of the
total-exact branch — it is forced for every `C_react^<=1` candidate that
successfully transfers and restores, while a broader P3 recode or lag can
diverge before that exact representation checkpoint (it must still solve the
next `sigma` pair). Any negative gadget offered against a candidate must be
FORCED on that candidate's own legal, `sigma`-consistent coupled history for
some legal S continuation; an arbitrary legal labeled real-board state (such
as S30) is insufficient by itself.

S23 says one turn-level reactive repair is too coarse. S29/S30 say two
physical F placements are an exact but nonautomatic service resource. The
sharp unresolved question is whether **one zero-lag, window-faithful recode
per single placement** can always be made genuine and strategy-consistent, or
whether some `sigma`-forced finite configuration demands more simultaneous
window certificates than any such recode can supply.

### 34.2 Provenance

**Input state.** Branch `hunt/gap-raw`, HEAD `5023169f`. This authoring pass
created no commit. **Reviewed/output artifact:** `8fb68864` (the commit the
R-ST4-REV hostile review examined; errata from that review are folded in
this file).

**Required documents read first, in order and in full.**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including its folded errata;
3. `STRATEGY_STEALING_REVIEW_ROUND2.md` (omitted from the original list in
   error — post-review erratum, Finding 12);
4. `STRATEGY_STEALING_ROUND3.md`, including section 24's folded errata; and
5. `STRATEGY_STEALING_REVIEW_ROUND3.md`.

Finding 10's exact older-`E_S` witness is carried into S25--S30 rather than
left as a generic terminal bullet. Finding 13's physical-persistence rule is
built into Definition 26.1 and S21. The per-synchronization wording correction
is used in S23.

**Rule sources checked.** The cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
All new rule uses are limited to radius-eight color-blind support, immediate
sequential updates, append-only forward histories, the three families of
length-six windows, and per-placement terminal detection.

**Machine work.** None. No Cargo command, Lean build, harness run, executable
proof search, or production-source edit was performed. No `GAP_RAW_*` file was
read or changed. The only intended deliverable written by this session is
`STRATEGY_STEALING_ROUND4.md`.

## 35. Post-review errata (R-ST4-REV, folded from STRATEGY_STEALING_REVIEW_ROUND4.md)

Hostile review of artifact `8fb68864` returned **SOUND-WITH-ERRATA**: neither
ranked theorem refuted. S23 (ranked (b), `C_react^<=1`) and S28 (ranked (c),
`C_shield`) both CONFIRMED-WITH-ERRATA; the rule-model citations, support
cut, deadline shield, S29 service criterion, S30 fork, and the six-blocker
hitting number all recompute exactly. The following repairs are folded in
place above:

1. **Finding 11 (MAJOR, folded in §34.1):** the resume checklist's "do one
   of six" connective was logically wrong — it mixed three alternative
   branches (A zero-lag repair / B shield-admissible lag / C same-step
   terminal certificate) with three mandatory obligations (4–6), and
   overstated (28.1) as forced on every dynamic candidate (it is conditional
   on the successful total-exact branch). §34.1 now carries the corrected
   quantifier structure, including the strategy-forced-reachability
   requirement on any negative gadget.
2. **Finding 3 (MINOR, folded in §28.1 argument):** the one-`Shat`-append
   fact is now derived from the immediate-transfer and real-turn premises
   (items 2/4), not phase equality alone.
3. **Finding 4 (MINOR, folded in §26.4):** candidate grammar (nonempty,
   with a legal partial-execution witness) vs extensional success class
   (proven empty by S23) are now distinguished; the round-3 comparison is
   scoped to the shared `q_1`-proxy S15 checkpoint.
4. **Finding 8 (MINOR, folded in §30.3 witness):** `C_shield` membership is
   defined as complete traces; the review's explicit terminating extension
   is recorded, making the class literally nonempty at that horizon.
5. **Finding 12 (MINOR, folded in §25.3/§32/§33/§34.2):** ledger/design-map
   scope repairs (S21 nonempty prefix; S22 premises on the P2/P6 rows;
   S29 fixed-`C_S,E_S`; P5R row acknowledges the S26 terminal-certificate
   alternative); `state.rs:149-160` + `283-288` citations added and the
   undo API relabeled as analysis-context; `STRATEGY_STEALING_REVIEW_ROUND2.md`
   restored to the required-corpus list; reviewed/output artifact recorded.

The review's ten-item corrected open agenda (its "Exact unresolved obstacles
after review") is adopted as the authoritative round-4 exit state; the named
resume gap remains `GAP-ZERO-LAG-WINDOW-RECODE / P5R-SERVICE` under the
corrected §34.1 quantifiers.
