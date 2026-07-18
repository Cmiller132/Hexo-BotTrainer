# Strategy stealing in engine Hexo, round 2: frontier-safe surplus and a fixed-map obstruction

**Worktree:** `hunt/gap-raw` at input HEAD `12980bc8`  
**Date:** 2026-07-17  
**Round-1 dependency:** `STRATEGY_STEALING_HEXO.md`, §§0–8  
**Ranked outcome:** **(b) achieved for the class defined in §12 [PROVEN].**  
**Global target:** `NL_F` remains **[OPEN]**.

The result is deliberately not an outcome theorem.  It excludes two natural
simulation families: every synchronous role-swapped shadow that maps
all real stones one-for-one (irrespective of its coordinate map), and the
second family that repairs the count mismatch by omitting stones but otherwise
uses a no-invention, fixed-isometry, immediate-copy history map.  Dynamic
recoding, virtual bookkeeping stones, and strategy-specific invariants that do
not promise a map on every legal history survive.

## 9. Statement boundary and inherited rule contract

### 9.1 Exact target and claim discipline

Let `F=Player0` be the compulsory opener and `S=Player1` the other player.  As
in round 1,

`NL_F : ∃ pure strategy σ_F, ∀ pure strategies σ_S, S never wins`.

Here “never wins” permits either a finite F win or an infinite history with no
six for either player.  The latter is a meta-level draw in the unbounded-game
idealization; it is not an engine `GameOutcome` value.

Every named or load-bearing proposition below is marked **PROVEN**,
**SKETCH**, **CONJECTURE**, **OPEN**, or **CITED**; explanatory prose inherits
the status and premises of the result it immediately follows.  There are no
machine-verified claims in this round.  No Cargo command, Lean build, harness,
or executable measurement was run.

### 9.2 Round-1 rules, unchanged [PROVEN]

This is the round-1 formalization, with the same notation and no rule variant.
Let

`H = {(q,r): q,r ∈ Z}`

with axial distance

`d((q,r),(q',r')) = max(|q-q'|, |r-r'|, |(q-q')+(r-r')|)`.

For a finite nonempty occupied set `O=X_F∪X_S`, put

`N_8(O)={c∈H : min_{z∈O} d(c,z)≤8}` and `L(O)=N_8(O)\O`.

The actual executable carrier is `i16`, not literal `Z²`; every finite gadget
in this document has coordinate magnitude at most 32 and is therefore inside
the same safe region used in round 1.  The distance formula is implemented at
`packages/hexo_engine/rust/src/coord.rs:76-82`; the carrier is declared at
`coord.rs:9-20`.

The only opening is `F@(0,0)`.  Along nonterminal play the ownership cadence is

`F ; S,S ; F,F ; S,S ; F,F ; …`.

The initial state and origin-only opening are at
`packages/hexo_engine/rust/src/state.rs:149-160` and
`packages/hexo_engine/rust/src/rules.rs:16-23`.  The phase transition is at
`state.rs:317-335`.  A normal coordinate is legal exactly when it is empty and
belongs to `L(O)` (`rules.rs:34-44`).  Radius eight is inclusive and
color-blind (`packages/hexo_engine/rust/src/coord.rs:84-95`,
`packages/hexo_engine/rust/src/legal.rs:17-18,123-145`, and
`packages/hexo_engine/rust/src/board.rs:91-95,167-170`).  The first placement
of a pair updates the board and legal store before the second is validated
(`board.rs:91-95`; `state.rs:293-335`), so a completed normal ordered turn
`(c_1,c_2)` is legal exactly when

`c_1∈L(O)`, `c_1` is nonwinning, and `c_2∈L(O∪{c_1})`.

Either player wins immediately after occupying all six cells of a window in
one of the unoriented directions `(1,0)`, `(0,1)`, `(1,-1)`.  The length and
directions are at `packages/hexo_engine/rust/src/tactics.rs:13-17,21-75`, the
all-six predicate at `tactics.rs:205-208`, and the per-placement update at
`tactics.rs:451-485`.  A win bypasses the phase change, so a win on the first
placement suppresses the second
(`packages/hexo_engine/rust/src/state.rs:302-337`).  `GameOutcome` has no draw
alternative and the transition has no
move-cap branch (`state.rs:64-71,289-337`).

These are the same radius-8 growth, 1-then-2:2 cadence, Maker–Maker six-in-a-row
win, and immediate termination rules proved in round 1.  S3 and S4 are used
but not reproved as broader claims: S3 kills identity/deletion legality at the
exact `(8,0)` query; S4 kills the direct one-deletion opening alignment.

### 9.3 Rule-level D6 symmetry [PROVEN]

Define

`ρ(q,r)=(-r,q+r)` and `κ(q,r)=(q,-q-r)`.

In cube coordinates `(q,r,s)` with `s=-q-r`, these send

`(q,r,s) ↦ (-r,-s,-q)` and `(q,r,s) ↦ (q,s,r)`.

They therefore preserve the maximum-absolute-coordinate distance, fix the
origin, and permute the unoriented axis set
`±{(1,0),(0,1),(1,-1)}`.  The twelve maps in
`D6=<ρ,κ>` preserve emptiness, `N_8`, ordered phases, six-windows, and the
winner.  Hence applying one `g∈D6` to every coordinate (including the stored
first coordinate in `SecondStone`) transports a legal full-game history to a
legal history with the same owner/phase sequence and outcome in the `Z²`
idealization, and on every executable history whose transformed arithmetic
stays inside the safe `i16` region.

This is a derived rule-equivariance theorem from `coord.rs:18-20,76-95`,
`legal.rs:123-145`, `rules.rs:16-44`, `tactics.rs:21-75,205-208`, and
`state.rs:149-160,302-337`; production code does not expose a named D6
operator.
Translations preserve normal-play distances and windows but are **not**
full-game symmetries because the opening is fixed at the origin.  Later uses
of `T=t∘g` explicitly choose `t` so that the selected shadow opening maps to
the origin; they do not translate an already rooted full game.

## 10. Exact surplus-stone calculus

Write `B_8(x)=N_8({x})` and, for arbitrary finite `C`, use the equivalent
union definition `N_8(C)=⋃_{c∈C}B_8(c)`, with `N_8(∅)=∅`.  Ownership is
intentionally absent from this section: the legal frontier is color-blind.
The universal lemmas in §§10–11 concern the declared `Z²` idealization; on the
literal carrier they apply wherever all displayed arithmetic stays safely
inside `i16`.

### 10.1 Reply lifting and the reverse frontier gap

**Lemma S5 (one-way legality monotonicity) [PROVEN].**  If `A⊆B` are finite,
nonempty occupied sets, `y∉B`, and `y∈L(A)`, then `y∈L(B)`.

*Proof.*  `y∈L(A)` gives `y∈N_8(A)`.  Since `A⊆B`,
`N_8(A)⊆N_8(B)`.  The separate hypothesis `y∉B` supplies emptiness. ∎

This is the useful half of “extra stones never hurt”: a shadow-prescribed
coordinate that is still empty on the larger real board remains legal there.
If the surplus already occupies that coordinate, legality does not transfer;
the coupling needs an own-stone filler argument, while an opponent-owned
collision is more serious.

**Lemma S6 (exact frontier-gap formula) [PROVEN].**  Let `A⊆B`, put
`E=B\A`, and consider only coordinates empty on the real board.  Then

`{y∉B : y∈L(B) and y∉L(A)} = (N_8(E)\N_8(A))\B`.          (S6)

*Proof.*  For `y∉B`, membership in `L(B)` is just membership in
`N_8(B)=N_8(A)∪N_8(E)`.  Failure of membership in `L(A)` is failure of
membership in `N_8(A)`, because `y∉B` implies `y∉A`.  Taking the indicated
set differences gives (S6). ∎

Call the right-hand side `Γ(A,E)`.  It is exactly the set of real-only moves
that a normalized subset shadow cannot copy when the real opponent is the next
actor.  The set itself is color-blind.  Thus surplus occupancy is
monotone-helpful for mapping a still-empty shadow reply to the real board, but
it is monotone-harmful for the reverse real-opponent projection precisely on
`Γ(A,E)`.  This separates the positive dominance fact from S3's trap.

### 10.2 Complete characterization for one surplus stone

**Theorem S7 (frontier-neutral surplus dichotomy) [PROVEN].**  Let
`x∈L(A)` and `B=A∪{x}`.  Exactly one of the following two mutually exclusive
cases applies; in each case the geometric condition is equivalent to the
displayed legal-set conclusion:

1. `B_8(x)⊆N_8(A)`, in which case
   `L(B)=L(A)\{x}` and deleting `x` preserves every real legal move; or
2. `B_8(x)⊄N_8(A)`, in which case
   `L(B)\L(A)=B_8(x)\N_8(A)≠∅`; every coordinate in that set is new and empty.

*Proof.*  Round-1 Lemma S1 gives

`L(A∪{x})=(L(A)\{x}) ∪ (B_8(x)\(A∪{x}))`.

Because `A⊆N_8(A)` and the legality of `x` gives `x∈N_8(A)`, every point of
`B_8(x)\N_8(A)` is automatically outside `A∪{x}`.  Conversely, a point in the
second S1 term is already in `L(A)` exactly when it lies in `N_8(A)`.  The two
cases follow. ∎

The exact **frontier-neutrality** condition is therefore not “the surplus is
ours”; it is the geometric containment `B_8(x)⊆N_8(A)`.  Under that condition
the extra stone consumes `x` and opens nothing.  Without it, deletion closure
fails on every coordinate in the displayed difference.  Cadence,
already-occupied prescriptions, and terminal correspondence remain separate
coupling obligations even in the neutral case.

**Corollary S7.1 (exposed surplus necessarily grows the frontier) [PROVEN].**
If, for one of the six signed cube-coordinate functionals, `x` is strictly
beyond every point of `A`, then `B_8(x)⊄N_8(A)`.

*Proof.*  For example, if `q(x)>max_{a∈A}q(a)`, take
`y=x+(8,0)`.  Then `d(x,y)=8`, while
`q(y)-q(a)≥9` for every `a∈A`, so `d(y,a)≥9`.  The other five cases are D6
images.  Apply S7. ∎

Round-1 S3 is an exposed-surplus instance: relative to
`A={(-8,0),(-16,0),(-24,0),(-32,0)}`, the extra `x=(0,0)` is strictly
q-extreme and `y=(8,0)` is in the new frontier.

### 10.3 Win detection alone really is monotone

**Lemma S8 (same-color six monotonicity) [PROVEN].**  If `X_p⊆X'_p` and
`X_p` contains a winning six-window for player `p`, then `X'_p` contains that
same window.

*Proof.*  A win is the inclusion of all six cells of one fixed window in the
player's set (`tactics.rs:205-208`).  Superset inclusion retains those cells. ∎

S8 is static and owner-sensitive.  S5–S7 are dynamic-legality statements and
owner-blind.  Combining S8 with only S5 does not make a strategy-stealing
coupling: real opponent moves in `Γ(A,E)` still lack a legal shadow image.

### 10.4 A global win-exact injection is already an isometry

Call an injection `f:H→H` **window-exact** when `f[W]` is an engine
six-window for every engine six-window `W`.  This is the natural global point
map condition if both colors' terminal sixes are to be represented literally,
rather than through proxy stones.

**Lemma S8.1 (window-exact injection rigidity) [PROVEN].**  Every global
window-exact injection has the form

`f(c)=t+g(c)`

for a translation vector `t` and some `g∈D6`.  In particular it is bijective
and preserves hex distance.  If it also commutes with every origin-centered
D6 map, then `t=0` and `g` is either the identity or the 180-degree rotation
`c↦-c`.

*Proof.*  Fix any grid line `p_n=p+nv` in one of the three engine directions
and let `W_n={p_n,…,p_{n+5}}`.  Injectivity gives

`|f[W_n]∩f[W_{n+1}]|=5`.

Two distinct engine windows on different axis lines meet in at most one cell;
two on one line meet in five cells exactly when they are consecutive
length-six intervals.  Hence all `f[W_n]` lie consecutively on one image line,
with each start shifting by `+1` or `-1`.  The sign cannot reverse: a reversal
would make `f[W_n]=f[W_{n+2}]`, whereas injectivity and
`|W_n∩W_{n+2}|=4` force their intersection to have size four.  The sign is
therefore constant.  Moreover,

`{p_n}=W_{n-5}∩W_{n-4}∩…∩W_n`.

The corresponding six consecutive image intervals also have a singleton
intersection, and that singleton advances by one image-line cell when `n`
advances.  Hence `f(p_n)` and `f(p_{n+1})` are adjacent.  This holds on every
grid line, so every adjacent pair maps to an adjacent pair.

At a vertex, the six neighbors have six distinct images among the six
neighbors of its image; the local map is onto.  Thus `f[H]` is neighbor-closed,
and connectedness of the hex grid makes `f` surjective.  It is a graph
automorphism.  The six neighbors of a vertex induce a 6-cycle, whose
automorphisms are D6.  Choose `g∈D6` agreeing with `f-f(0)` on the origin's
six neighbors, and normalize by composing with its inverse.  If the normalized
map fixes a vertex and its six neighbors, then for each neighbor it also fixes
the old vertex and the two other common neighbors of their edge (the edge lies
in exactly two unit triangles).  Those three fixed cells force the induced
automorphism of the neighbor's 6-cycle to be the identity.  Thus its whole
neighbor set is fixed.  Connectivity propagates this from the origin to every
vertex, giving `f(c)=f(0)+g(c)` before normalization.

Finally, commuting with all origin-centered D6 maps forces `f(0)` to be fixed
by all of D6, hence `f(0)=0`, and forces `g` into the center of D6.  That center
is `{id,-id}`. ∎

**Consequence of S8.1 [PROVEN].**  Section 12 may equivalently replace its
explicit isometry assumption by “the retained coordinate encoder is the
restriction of one global window-exact injection.”  Finite, history-specific
encodings or proxy-assisted terminal maps are not covered by S8.1.

## 11. Cadence arithmetic for every synchronous role swap

This section sharpens S4 from the particular deletion of the opening stone to
an exact parity law.

**Cadence count identities [PROVEN].**  At the real `F FirstStone` checkpoint
immediately after S has completed its `k`-th normal turn (`k≥1`), the
real-label counts are

`(|X_F|,|X_S|)=(2k-1,2k)`.                              (11.1)

A legal role-swapped shadow checkpoint at which F, now the shadow second
player, is at `FirstStone` has shadow-role counts

`(shadow opener S, shadow second F)=(2j+1,2j)`            (11.2)

for some `j≥0`.  Similarly, at a real `S FirstStone` checkpoint after both
players have completed `k` normal turns, real counts are `(2k+1,2k)`, while a
legal shadow checkpoint with S, the shadow opener, at `FirstStone` has counts
`(2j+1,2j+2)`.

**Theorem S9 (equal-odd deletion law) [PROVEN].**  Consider a no-invention,
owner-faithful role-swapped projection that represents retained stones
one-for-one, without duplication, at either kind of same-actor `FirstStone`
checkpoint.  Let `a` be the number of real F stones omitted and `b` the number
of real S stones omitted.  If the retained stones form a legal shadow
checkpoint, then

`a=b=2(k-j)-1`,

so the two omitted counts are equal, positive, and odd.  The smallest possible
alignment omits exactly one stone of each real color.

*Proof.*  At the F checkpoint, after omission and role swap the shadow counts
are `(2k-b,2k-1-a)`.  Equating these with (11.2) gives

`b=2(k-j)-1=a`.

Nonnegative omission forces `j≤k-1`, so this common number is a positive odd
integer.  At the S checkpoint, equating `(2k-b,2k+1-a)` with
`(2j+1,2j+2)` gives the identical equations. ∎

**Corollary S9.1 (total pointwise embeddings cannot synchronize) [PROVEN].**
No same-phase role-swapped shadow obtained by mapping **every** current real
stone one-for-one, with no invented shadow stones, can be a legal `FirstStone`
checkpoint for the corresponding actor.

*Proof.*  At every checkpoint covered by S9, a total one-for-one
representation has `a=b=0`, contradicting S9.  At the only earlier normal
checkpoint, immediately after the compulsory F opening, role swap would give
shadow counts `(opener S,second F)=(0,1)`, but a legal full game cannot contain
a second-player stone before its opener. ∎

This corollary is independent of geometry.  In particular it covers every
total injective real-to-shadow coordinate map, including any map commuting
with D6, even if that map is nonlinear.  “Pointwise-injective” here means a
total one-for-one representation of all real stones; it does **not** mean the
opposite-direction inclusion of a smaller shadow board into a larger real
board.  The latter is possible and is exactly the setting of Lemma S5.

S9 also shows why deleting only the compulsory F stone cannot repair S4: the
missing real-S count has to be the same odd number.  At the first possible
alignment, one F and one S stone must be omitted.  Coordinate translation,
rotation, or reflection cannot alter this arithmetic.

## 12. Ranked outcome (b): the synchronous isometric class is impossible

### 12.1 The class being excluded

Define `C_iso` to be the following class of universal history maps after the
first possible repaired synchronization.  A member of `C_iso` promises, at
every legal nonterminal real `S FirstStone` checkpoint beginning after F has
completed its first normal pair:

1. a legal full-game shadow prefix at S's role-swapped opener `FirstStone`
   checkpoint (real S is the shadow opener; real F is the shadow second
   player);
2. **no invention and owner fidelity:** there is an owner-respecting subset
   `R` of real stones and the shadow occupancy is exactly `T[R]` for the map
   in item 3, with roles swapped; the map may omit real stones but may not
   duplicate them, add proxy stones, or recolor a retained stone;
3. **isometric point map:** at the checkpoint, all retained coordinates are
   mapped by one `T=t∘g`, with `g∈D6`; `T` may be chosen from the history, and
   retained placements may be reordered if that helps legality; and
4. **immediate fixed-map opponent copy:** the next real placement by the
   actor at that checkpoint is appended at `T(y)` in the shadow, using the
   same `T`.

The promise is universal over legal input histories, rather than only over
histories generated by a particular candidate strategy.  This template
subsumes as candidate constructions the identity/deletion map, every fixed
translated/rotated/reflected version, and the delayed/translated
minimum-deletion repair examined by the round-3 review.  Allowing arbitrary
reorder in item 3 makes the negative theorem stronger, not weaker.

### 12.2 Exact legal witness, replayed

Use the round-1 coordinates

`x=(0,0), u=(-8,0), v=(-16,0), w=(-24,0), z=(-32,0), y=(8,0)`

and the real placement sequence

`F@x ; S@u,S@v ; F@w,F@z ; S@y`.                         (12.1)

**Lemma S10 (real witness legality and nonterminality) [PROVEN].**  Every
placement in (12.1) is legal, including both within-turn second placements,
and no displayed prefix is terminal.

*Proof.*  `x` is the compulsory opening.  The five normal placements have the
following respective supports:

`u←x`, `v←u`, `w←v`, `z←w`, `y←x`,

all at exact distance eight.  Every coordinate is new.  The phase sequence is
`F Opening; S First; S Second; F First; F Second; S First`.  Through `y`, each
player has only three stones, so neither can contain a six-window.  This is the
accepted round-1 S2 replay, checked directly against §9.2. ∎

Immediately before `S@y`, the real position is a legal `S FirstStone`
checkpoint with real counts `(F,S)=(3,2)`.

### 12.3 Exhaustive shadow alignment and failure

**Theorem S11 (`C_iso` obstruction) [PROVEN].**  No member of `C_iso` exists.
More locally: on the legal prefix of (12.1) through `F@z`, every no-invention,
same-phase, role-swapped isometric shadow is forced to delete `F@x` and `S@u`
and retain `S@v,F@w,F@z`; under every permitted `T`, the next legal real move
`S@y` maps to an illegal shadow coordinate.

*Proof.*  A legal shadow with its opener S at its first normal `FirstStone`
checkpoint has counts `(S,F)=(1,2)`.  No later such checkpoint fits inside the
available `(2,3)` stones.  Therefore any item-1/item-2 shadow must choose one
of `{u,v}` as its opening and two of `{x,w,z}` as the shadow-second pair.  This
also follows from S9: exactly one stone of each color must be omitted.

Because `T` preserves distance, it suffices to exhaust the untransformed
coordinates.

- If `u` is the shadow opening, the distances from `u` to `x,w,z` are
  `8,16,24`.  Only `x` can be the pair's first placement.  After `x`, neither
  `w` nor `z` is legal: their distances from the occupied set `{u,x}` are
  respectively `min(16,24)=16` and `min(24,32)=24`.
- If `v` is the shadow opening, the distances from `v` to `x,w,z` are
  `16,8,16`.  Hence `w` is the only possible first placement.  Then `z` is
  legal from `w` at distance eight, whereas `x` remains at distances 16 and
  24 from `{v,w}`.

Thus the only legal selection and order are opening `v`, followed by the pair
`w,z`.  Necessarily `T(v)=(0,0)` for the shadow opening.  This shadow prefix is
indeed legal: `d(v,w)=d(w,z)=8`, and with only one versus two stones it is
nonterminal.

The real continuation `y` is legal solely through the omitted `x`:

`d(y,x)=8`, while `d(y,u),d(y,v),d(y,w),d(y,z)=16,24,32,40`.

The retained shadow occupancy is `T({v,w,z})`; isometry gives distances
`24,32,40` from `T(y)`.  Hence `T(y)∉L(T({v,w,z}))`.  Item 4 fails.  Since the
selection was exhaustive and `T` was arbitrary, no `C_iso` map can keep its
universal promise. ∎

**Corollary S11.1 (not an orientation artifact) [PROVEN].**  The same failure
holds after any D6 rotation or reflection of the real witness and for every
history-dependent choice of translation composed with D6 at this checkpoint.

*Proof.*  §9.3 transports real legality, phase, and nonterminality under D6;
the proof of S11 uses only distances, owners, counts, and the requirement that
the chosen opening map to the origin. ∎

S11 strictly extends the reviewed example in scope: it does not merely test
the particular translation `+(16,0)`.  It proves that the translation was
forced up to D6 after an exhaustive choice of retained stones, and every such
isometry loses `y`.

### 12.4 Exact survivor boundary [PROVEN]

S9 and S11 together prove the ranked-(b) result for a well-defined class:

- retain every real stone one-for-one: cadence makes synchronization
  impossible, regardless of the coordinate map;
- omit the stones forced by cadence but otherwise use a no-invention
  translation/D6 point map and copy the next opponent move immediately: S11
  makes the shadow continuation illegal.

Therefore any surviving simulation must violate at least one defining
condition: it must introduce justified proxy bookkeeping, use a genuinely
non-isometric or move-dependent recoding, change the lag/phase relation,
decline to copy the opponent immediately, or prove an invariant only for its
own strategy-generated histories that excludes (12.1).  This is an exact
boundary for `C_iso`, not a theorem against all non-identity simulations and
not an outcome result.

## 13. A separate obstruction to naive positive time lag

A one-turn lag can repair counts on paper by asking the shadow strategy for a
reply before the real opponent has made the intervening turn.  Pure-strategy
perfect information makes that reply available to the opponent as well.

**Theorem S12 (preannounced-first-coordinate collision) [PROVEN].**  Suppose a
time-lagged exact-copy coupling fixes its next real F **first** coordinate `r`
before an intervening S turn, with the current real state legal, nonterminal,
and at `S FirstStone`.  Suppose also that, after normalizing the shadow
coordinates, its occupied shadow support `A` is a subset of the current real
occupancy `B`, and that `r∈L(A)` is still empty in `B`.  Then S has a legal
response that destroys the promised exact copy: S plays `r` as its first
coordinate.

*Proof.*  Lemma S5 gives `r∈L(B)`, so the move is legal.  If it completes an S
six, the real game has already ended against F.  Otherwise `r` becomes
S-occupied and the engine's occupied-cell rule (`rules.rs:34-44`) forbids F
from realizing the preannounced copy on its next turn.  S can complete its
current pair: after any nonwinning first placement, take an occupied cell of
maximal q-coordinate and its empty outward `(1,0)` neighbor, which is legal at
distance one in `Z²`. ∎

The counterstrategy computes `r` from the fixed pure coupling and history;
“preannounced” means causally fixed before S chooses, not that a private
internal value is communicated by an extra game action.

S12 concerns a first coordinate legal before the intervening turn.  It does
not concern a pair's second coordinate that may become legal only through its
first, an already F-owned prescription, or a reply chosen adaptively after S
has acted.  Consequently it rules out the naive exposed lag, not every
time-shifted coupling.  A survivor needs a collision-safe recoding theorem or
must defer the shadow query until after the real opponent move.

### 13.1 The FIFO one-stone lag still meets S3

Consider the natural opening repair that chooses one of S's first two stones
as the shadow opening, queues the other, realizes a shadow-F pair, and then
uses the queued S stone together with S's next real placement as the shadow
opener's first ordinary pair.

**Theorem S13 (one-stone FIFO frontier obstruction) [PROVEN].**  No universal
exact-copy implementation of the following one-sided schedule, using one
translation/D6 isometry, can handle every legal history: the real opening `x`
remains omitted; exactly one of the first real S pair is the shadow opening;
the other is the sole FIFO-queued stone; the next real F pair is copied as the
shadow F pair; and the queued S stone must be paired with S's next real
placement.  One checkpoint isometry `T` maps the selected opening, that F
pair, and the queued/new S pair; no proxy or other represented support is
allowed.

*Proof.*  Play the legal real history

`F@x ; S@a,S@b ; F@p,F@q ; S@c`,                       (13.1)

where

`x=(0,0), a=(-8,0), b=(-16,0), p=(-12,4),`
`q=(-12,12), c=(8,0)`.

Legality is exact: `d(a,x)=8`, `d(b,a)=8`,
`d(p,a)=4`, `d(p,b)=8`, `d(q,p)=8`, and `d(c,x)=8`.
All cells are new, the ownership/phase sequence is the ordinary
`F;S,S;F,F;S`, and through `c` each player has only three stones.  Thus (13.1)
is legal and nonterminal through every prefix.

Either `a` or `b` may be selected as the shadow S opening; queue the other.
The pair `p,q` is legal after either choice, because `p` is at distance four
from `a` and eight from `b`, and `q` is at distance eight from `p`.  At the
next shadow-S pair, the queued stone is legal at distance eight from the
chosen opener.  But

`d(c,a),d(c,b),d(c,p),d(c,q)=16,24,20,20`.

If `c` is ordered first, it is illegal before the queue is placed.  If the
queued stone is ordered first, `c` is still farther than eight from every
represented stone and is illegal second.  A translation composed with D6
preserves all these distances.  The real `c` was legal solely through the
omitted compulsory opener `x`, so both FIFO orders fail. ∎

S13 allows either choice of first-turn S stone and either ordering of the
queued/new S pair.  It does not cover a queue with proxy support, a non-isometric
encoding of `c`, or a coupling that changes more of the shadow schedule.

### 13.2 A literal one-stone lag also loses terminal fidelity

**Theorem S14 (one-lag terminal-count obstruction) [PROVEN].**  A universal,
owner-faithful pointwise simulation that always leaves one real S stone
unrepresented, makes every shadow stone the single image of one represented
real stone, adds no proxy stones and no duplicated or recolored images, and
requires real S wins occurring before any real F win to map no later to
terminal shadow histories cannot preserve all legal histories.

*Proof.*  Use

`F@(0,0); S@(1,0),S@(2,0); F@(0,8),F@(0,16);`

`S@(3,0),S@(4,0); F@(0,24),F@(0,32); S@(5,0),S@(6,0)`.

Every coordinate is new.  The S placements grow consecutively along the
q-axis.  The F pairs are legal via the distance-eight chain
`(0,0)→(0,8)→(0,16)→(0,24)→(0,32)`, including both within-turn second
placements.  No earlier S prefix has six stones.  The final placement
`S@(6,0)` completes the engine window `{(1,0),…,(6,0)}`.  F has only five
stones, spaced eight apart after the origin, and has no six.

At that terminal checkpoint a literal one-S-stone-lag shadow represents at
most five S stones.  It also has at most the five literal F stones.  With no
proxy, duplication, or recoloring, neither shadow color can own a six-window.
The real S-terminal history, with no earlier F win, therefore maps to a
nonterminal shadow history, contrary to the stated loss-reflection promise. ∎

S14 is a universal-history-map obstruction, not an assertion about histories
generated by an assumed winning strategy.  A coupling may survive it only by
proving that this real history is unreachable under its own strategy, by
catching the real win with an earlier mapped F win, or by giving nonliteral
proxy semantics a separate correctness proof.

## 14. The non-loss determinacy bridge

Round 1 correctly left a logical gap between “S has no winning strategy” and
the requested existence of one F non-losing strategy.  For the declared
unbounded idealization, open determinacy closes that gap.

### 14.1 Exact external theorem

**Gale–Stewart open determinacy [CITED].**  Every two-player game of perfect
information `G(A;T)` whose payoff `A` is open in the branch space of the game
tree is determined: one of the two players has a pure winning strategy.

The exact class and statement are Theorem 1.2.4 (“All open games are
determined”) in Donald A. Martin, *Determinacy of Infinitely Long Games*,
§1.2, pp. 12–15, which attributes the theorem to Gale and Stewart (1953):
<https://www.math.ucla.edu/~dam/booketc/D.A._Martin%2C_Determinacy_of_Infinitely_Long_Games.pdf>.
The original source is David Gale and F. M. Stewart, “Infinite Games with
Perfect Information,” *Contributions to the Theory of Games II*, Annals of
Mathematics Studies 28 (1953), pp. 245–266:
<https://doi.org/10.1515/9781400881970-014>.

Only open (`Σ⁰₁`) determinacy is invoked; no claim requiring full Borel
determinacy or an additional determinacy axiom is used.

### 14.2 Encoding Hexo as an open game

**Lemma D1 (Hexo S-win is an open perfect-information payoff) [PROVEN].**
After the forced F opening, the unbounded Hexo idealization is representable as
an alternating, countable-alphabet, finitely branching perfect-information
game whose S-win payoff is open.

*Proof.*  Group each ordinary turn into one macro-action.  A macro-action is
either:

- a legal singleton `c_1` that wins immediately, or
- an ordered pair `(c_1,c_2)` where `c_1` is legal and nonwinning and
  `c_2∈L(O∪{c_1})`.

This grouping is exact because a first-placement win suppresses the second,
while every nonwinning first placement must be followed by the same player's
second placement (§9.2).  After the forced opening, macro-turns alternate S,
F, S, F, and both players see the full finite history.

The strategy models are equivalent.  No opponent action occurs between
`c_1` and `c_2`, and the successor after `c_1` is deterministic.  Therefore a
pure single-placement strategy induces exactly one macro choice—singleton if
`c_1` wins, otherwise its ordered pair—and every legal macro strategy expands
back into the corresponding two sequential prescriptions.  Reattaching F's
forced opening gives strategies with the quantifiers used in `NL_F`.

The coordinate and pair alphabet is countable.  At every finite position the
legal set is finite: a radius-eight hex ball has
`1+3·8·9=217` cells, so with `n≥1` occupied cells,
`|L(O)|≤217n-n=216n`; the second coordinate is similarly bounded after one
more placement.  This finiteness also follows directly from the finite-radius
enumerator and stored legal-set iteration at `coord.rs:84-95`,
`legal.rs:44-50,75-112`, and `state.rs:203-252`.

There is no nonterminal dead end in the `Z²` idealization.  For finite
nonempty `O`, choose an occupied cell with maximal q-coordinate.  Its outward
neighbor one step in the `(1,0)` direction has larger q, hence is empty, and
is legal at distance one.  The same argument after a nonwinning first
placement supplies at least one legal second placement.  This is an
idealized-board statement, not a claim about arithmetic at an `i16` boundary.

Pad a finite terminal play by a forced dummy symbol forever.  Infinite
nonterminal histories remain infinite branches.  Let `W_S` be the branches on
which S completes six at some finite prefix.  If a branch is in `W_S`, the
finite prefix ending at that placement determines membership for every
extension, so `W_S` is a union of basic cylinders and is open. ∎

**Theorem D2 (non-loss bridge) [PROVEN from the CITED theorem].**  In the
unbounded Hexo idealization,

`NL_F  ⇔  S has no winning strategy`.

*Proof.*  Apply Gale–Stewart open determinacy to `W_S`, with S as the first
macro-player after the forced opening and F as the second.  Either S has a
strategy forcing membership in `W_S`, or F has a strategy forcing its
complement.  The complement consists exactly of plays in which S never wins:
either F wins at a finite prefix or play is infinite with neither color ever
completing six.  The latter is the declared meta-level draw.  Hence the second
alternative is precisely `NL_F`.  The two alternatives cannot both hold,
because playing the two purported strategies against one another would give
one branch both in and outside `W_S`. ∎

Thus **GAP-NONLOSS-DETERMINACY is discharged [PROVEN from CITED]**.  A future
stealing proof need only refute the existence of an S winning strategy; it
does not additionally need to construct F's non-losing strategy explicitly.
This does not itself refute S's winning-strategy alternative and therefore
does not prove `NL_F`.

## 15. Result ledger, survivor class, and resume point

| Claim | Status | Exact scope |
|---|---|---|
| Inherited rule contract | **PROVEN** | Production engine in the safe carrier region; `Z²` only for the declared infinite-play idealization |
| Rule-level D6 equivariance | **PROVEN** | Origin-fixing rotations/reflections; translations excluded as full-game symmetries |
| S5 reply lifting | **PROVEN** | Shadow occupancy subset of real occupancy; requested coordinate still real-empty |
| S6 exact frontier gap | **PROVEN** | Every finite `A⊆B`, on coordinates empty in `B` |
| S7 frontier-neutral iff condition | **PROVEN** | One legal surplus `x`; neutrality iff `B_8(x)⊆N_8(A)` |
| S7.1 exposed surplus growth | **PROVEN** | Any one-surplus strict extreme in a signed cube direction |
| S8 same-color win monotonicity | **PROVEN** | Static six detection only |
| S8.1 window-exact injection rigidity | **PROVEN** | Global injection mapping every six-window to a six-window |
| S9 equal-odd deletion law | **PROVEN** | Same-actor, same-`FirstStone`, no-invention role swap |
| Total one-for-one/D6-equivariant shadow | **PROVEN** | Corollary S9.1 proves impossibility; geometry cannot repair cadence counts |
| S10 accepted witness replay | **PROVEN** | Six listed placements; exact supports and nonterminality |
| S11 `C_iso` obstruction | **PROVEN** | Universal, no-invention, same-phase, translation/D6, immediate-copy maps |
| S11.1 D6 witness orbit | **PROVEN** | Every rotation/reflection and checkpoint translation choice |
| S12 preannounced lag collision | **PROVEN** | Empty first coordinate exposed across an intervening S turn |
| S13 one-stone FIFO frontier repair | **PROVEN** | Impossibility of the exact-copy one-queue scheme on the displayed legal history |
| S14 one-lag terminal fidelity | **PROVEN** | Impossibility for literal no-proxy universal history maps |
| Gale–Stewart open determinacy | **CITED** | Open payoffs on two-player perfect-information game trees |
| D1 Hexo S-win payoff is open | **PROVEN** | Macro-turn encoding of the unbounded idealization |
| D2 `NL_F ⇔` no S winning strategy | **PROVEN from CITED** | Pure strategies; finite F win or infinite draw is F's complement payoff |
| `NL_F` | **OPEN** | This round gives no contradiction to an arbitrary S winning strategy |
| GAP-OPENING-ALIGNMENT beyond the excluded classes | **OPEN** | Proxy, non-isometric, or more complex lag schemes survive |
| GAP-FRONTIER-COUPLING beyond the excluded classes | **OPEN** | Must encode `Γ(A,E)` moves and preserve ordered placement/terminal timing |

There are no **SKETCH** or **CONJECTURE** claims in the result ledger.  The
ranked-(b) result is S9+S11: it proves an obstruction to the explicitly defined
class, not to all conceivable non-identity simulations.  S5–S8 are the honest
positive partial result: surplus stones preserve same-color wins and lift
still-empty shadow replies, while S6/S7 identify exactly when they create an
reverse-projection frontier gap (a real-opponent gap in the coupling use).

The surviving design space is now precise.  A successful coupling must use at
least one of the following mechanisms and prove its cost:

1. proxy or virtual shadow stones with a terminal-faithfulness theorem;
2. dynamic, non-isometric recoding of frontier-gap moves and shadow replies;
3. a more complex two-sided lag whose prescriptions are not exposed to S and
   whose queue cannot hide an S six; or
4. a strategy-specific invariant proving that its own real F moves keep every
   surplus frontier-neutral (`Γ(A,E)=∅`) or make the obstruction histories
   unreachable.

**Named refined resume point — `GAP-FRONTIER-COUPLING / GAP-PROXY-SHADOW` [OPEN].**
Construct or obstruct a causal proxy scheme that (i) accounts for
the one-F/one-S discrepancy identified by S9, whether by omissions, proxies,
or another justified encoding, (ii) gives every move in `Γ(A,E)` a legal
shadow representation, (iii) maps the shadow strategy's two sequential
replies back after observing S so S12 cannot preempt them, and (iv) preserves
both finite-win timing and the ordered `FirstStone`/`SecondStone` phase.
S13's `(8,0)` move and S14's final six are the first mandatory tests.

## 16. Provenance

**Input state.** Branch `hunt/gap-raw`, HEAD `12980bc8`.  This authoring pass
created no commit.

**Read first, in order.** `STRATEGY_STEALING_HEXO.md`, then the stealing
verdict in `GAP_RAW_REVIEW_ROUND3.md`.  The review's accepted S3/S4 scope was
preserved: neither obstruction was inflated into an outcome theorem.

**Rule sources read.** The `coord.rs`, `legal.rs`, `rules.rs`, `state.rs`,
`tactics.rs`, and `board.rs` files under `packages/hexo_engine/rust/src/`.
All new rule facts are tied to exact source ranges in §§9 and 14.  The D6
statement is derived from those production predicates; the engine has no
named D6 transform.

**External theorem source.** Gale–Stewart open determinacy, cited exactly in
§14.1 and instantiated only for the open payoff “S completes six at a finite
prefix.”

**Machine work.** None.  No Cargo command, Lean build, harness run, or source
edit was performed.  The only intended deliverable written by this session is
`STRATEGY_STEALING_ROUND2.md`.
