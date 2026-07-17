# Strategy stealing in engine Hexo under growth-constrained legality

**Worktree:** `hunt/gap-raw` at input HEAD `283348dce09d42b67e364e0b2f2b63166b6b5f4d`  
**Date:** 2026-07-17  
**Disposition:** the implemented rules are tied down **[PROVEN]**; the ordinary
delete-the-extra-stone strategy-stealing simulation is not a legal-history
simulation **[PROVEN]**; “the opener has a non-losing strategy” remains
**[OPEN]**.

Throughout, **opener** means the human first player, called `Player0` by the
zero-indexed engine. The other player is the human second player, called
`Player1` by the engine. Thus the requested “Player 1” non-loss statement is
about engine `Player0`, not engine `Player1`.

## 0. Verdict and statement boundary

**Rules verdict [PROVEN].** The actual engine is a Maker–Maker game on the axial
hex grid. Engine `Player0` makes the compulsory singleton opening at `(0,0)`.
Engine `Player1` then takes the first ordinary turn, and every ordinary turn is
an ordered pair of sequential placements. A normal placement is legal exactly
when it is empty and within hex distance eight of at least one stone of either
color. The legal zone is updated after the first placement of a pair, so the
second placement may use the first as its sole growth support. Either player
wins immediately upon completing six consecutive stones on one of the three
axes; this is checked after each single placement, so a first-placement win
ends the turn. There is no finite draw result or move cap in the rules.

**Stealing verdict [PROVEN].** The general deletion-closure lemma required by
the classical shadow construction is false for Hexo. An exact legal prefix
below contains an opponent move `(8,0)` whose only radius-eight support is the
compulsory opening stone `(0,0)`. Erasing `(0,0)` makes that real opponent move
illegal in the resulting normal-placement geometry. This does not purport to
align a complete classical steal from the actual opening—the independent
cadence obstruction in §5 shows that alignment already fails—but it proves
that any repaired shadow must do more than delete one unmatched own stone and
copy later moves.

**Non-loss target [OPEN].** This obstruction invalidates that proof, not the
theorem it was intended to prove. No strategy for the opener is constructed
here, and no strategy for the second player is constructed. In particular,
this document does **not** prove that an extra own stone is strategically
harmful; it proves the narrower board-level fact that the normal-placement
legal-set predicate is not deletion-monotone.

## 1. Source tie-down

### 1.1 Which crate owns the rules [PROVEN]

Live authoritative transitions used by `hexfield_eq` come from
`hexo_engine`. Its dependency is
`hexo_engine.workspace = true` (`packages/hexfield_eq/Cargo.toml:22-24`), which
resolves to the local crate at `packages/hexo_engine` (`Cargo.toml:15-18`). Its
Rust intake layer imports `hexo_engine::HexoState` directly
(`packages/hexfield_eq/rust/src/state.rs:15`) and clones live engine states
through the engine capsule (`packages/hexfield_eq/rust/src/state.rs:27-39` and
`:63-90`). The accelerator's module header says the same thing
(`packages/hexfield_eq/rust/src/lib.rs:1-4`). Therefore the authoritative rule
symbols for this question are in `packages/hexo_engine/rust/src/`.

### 1.2 Coordinate carrier caveat [PROVEN]

The rule contract calls the grid unlimited/infinite
(`packages/hexo_engine/rust/src/coord.rs:1-4` and
`packages/hexo_engine/rust/src/board.rs:1-5`), while the executable coordinate
carrier stores `q,r` as `i16` (`coord.rs:9-16`). The mathematical formulation
below uses the declared unbounded rule idealization. Every coordinate in the
obstruction has absolute value at most 32, so the proof is valid both for that
idealization and on the literal safely represented region; it makes no claim
about arithmetic or overflow behavior at an `i16` boundary. Auxiliary hunt
and reference modules elsewhere in `hexfield_eq` model rules for testing and
proof search, but they are not the live production transition authority.

## 2. The game as implemented

### 2.1 Board, distance, and windows [PROVEN]

Let

`H = {(q,r): q,r ∈ Z}`

with axial distance

`d((q,r),(q',r')) = max(|q-q'|, |r-r'|, |(q-q')+(r-r')|)`.

This is the formula implemented by `hex_distance`
(`packages/hexo_engine/rust/src/coord.rs:76-82`). A finite board occupancy is a
pair `(X_0,X_1)` of disjoint finite subsets of `H`, one per player. The sparse
board stores a coordinate-to-owner map and an insertion-ordered occupied list
(`packages/hexo_engine/rust/src/board.rs:18-29`).

The three unoriented line directions are

`v_Q=(1,0)`, `v_R=(0,1)`, and `v_QR=(1,-1)`.

They are the engine's `Axis::{Q,R,QR}` vectors
(`packages/hexo_engine/rust/src/tactics.rs:21-52`). For `a∈H` and one of those
directions, a winning window is

`W(a,v)={a,a+v,a+2v,a+3v,a+4v,a+5v}`.

The window length is exactly six (`tactics.rs:13-17`), and
`WindowEntry::is_win_for` requires all six cells to belong to one player
(`tactics.rs:205-208`).

### 2.2 State and phase [PROVEN]

A nonterminal engine state contains `(X_0,X_1,p,φ,n)`, together with history
and caches, where `p` is the player making the next single placement,
`n=|X_0|+|X_1|`, and

- `φ=Opening`;
- `φ=FirstStone`; or
- `φ=SecondStone{first=c_1}`.

The actual enum and stored first coordinate are at
`packages/hexo_engine/rust/src/state.rs:46-56`; the full state fields are at
`state.rs:95-112`.

The initial state is empty, has `p=Player0`, and is in `Opening`
(`state.rs:149-160`). The only legal opening is `Player0@(0,0)`
(`packages/hexo_engine/rust/src/rules.rs:11-23`). If it does not win (a
singleton cannot), control passes to `Player1` at `FirstStone`. A nonwinning
`FirstStone` placement keeps the same player and changes the phase to
`SecondStone`; a nonwinning `SecondStone` placement passes to the other player
at `FirstStone` (`state.rs:317-335`). Consequently the ownership sequence is

`Player0 ; Player1,Player1 ; Player0,Player0 ; Player1,Player1 ; …`.

This singleton-plus-pairs cadence is part of the game; it is not an encoding
convention.

### 2.3 Normal placement legality and within-turn growth [PROVEN]

For nonempty occupied set `O=X_0∪X_1`, define

`N_8(O)={c∈H : min_{z∈O} d(c,z)≤8}`

and

`L(O)=N_8(O)\O`.

At either normal phase, the legal coordinate set is exactly `L(O)`. The rule
first rejects occupied cells and then tests membership in the incremental
legal store (`packages/hexo_engine/rust/src/rules.rs:34-44`). That store has
radius constant eight and, after every placement `x`, inserts every empty cell
within distance eight of `x` (`packages/hexo_engine/rust/src/legal.rs:17-18,
123-145`). Ownership is not consulted. `SecondStone` additionally records and
rejects reuse of the first coordinate (`rules.rs:24-30`), although ordinary
occupancy already rejects it.

Thus an ordinary ordered turn `(c_1,c_2)` is legal precisely when

`c_1∈L(O)` and `c_2∈L(O∪{c_1})`.

The second condition is genuinely sequential: board placement updates the
legal store before the phase transition
(`packages/hexo_engine/rust/src/board.rs:82-105`; `state.rs:302-317`).

### 2.4 Winning, termination, and the meaning of non-loss [PROVEN]

After every single placement, the engine updates the eighteen incident
length-six windows and checks `WindowUpdate::has_win`
(`packages/hexo_engine/rust/src/tactics.rs:1-5,451-485` and
`:318-333`). If true, it records the
current player as winner and does not advance player or phase; otherwise it
advances the phase machine (`packages/hexo_engine/rust/src/state.rs:309-337`).
Hence both colors are Makers, and a win on the first stone of a normal turn
suppresses the second stone (`state.rs:3-10`). Terminal states expose no legal
moves (`state.rs:203-252`), and attempted post-terminal placement is rejected
(`packages/hexo_engine/rust/src/rules.rs:11-14`).

`GameOutcome` contains a winner and placement count, with no draw variant; its
source comment states that Hexo has no normal draw under the current rules
(`state.rs:64-71`). There is also no placement-limit terminal branch in
`apply_with_delta`. For the declared unbounded-game idealization studied here,
an infinite legal history in which neither player completes six is called a
*draw*. This is a meta-definition, not a literal `GameOutcome` variant. A
strategy is *non-losing* when every opponent continuation either gives its
owner a finite win or produces such an infinite draw. The engine state itself
has no explicit move-cap branch; an external runner cap, if one is imposed, is
not a game rule or a draw result.

Define the requested target formally as

`NL_F : ∃ pure strategy σ_F, ∀ pure strategies σ_S, S never wins`,

where `F=Player0` is the opener and `S=Player1`. The strategy's actions are
single placements, so it must specify both coordinates of each normal turn
sequentially, including histories on which its first coordinate changes the
legal set for its second.

## 3. What the classical stealing proof needs

### 3.1 The deletion-shadow step [PROVEN]

In the usual unrestricted strong positional game, the contradiction argument
starts by assuming that the second player has a winning strategy `σ`. The
opener takes one arbitrary cell `x`, then runs `σ` in a shadow history obtained
by deleting `x` and exchanging roles. Two closure facts make that construction
well-defined:

1. every opponent move legal on the real board is also legal in the shadow
   board; and
2. if `σ` asks for `x`, the real player already owns it and may use an arbitrary
   otherwise legal filler without losing a static winning set.

The first fact is the one relevant here. With unrestricted placement, adding
an own stone removes `x` from the opponent's options but creates no new option,
so a real opponent history projects to a legal shadow history. Only after that
projection is established is it meaningful to query `σ`.

For Hexo, call the corresponding required property **deletion closure**:

> For occupied set `O`, extra stealing-player stone `x`, and every opponent
> coordinate `y`, if `y∈L(O∪{x})` and `y≠x`, then `y∈L(O)`.

The identity/deletion version of this property is false. Static six-in-line
monotonicity does not repair it: adding an own stone cannot erase an already
owned six, but strategy stealing needs a statement about the opponent's legal
continuations before either player has won.

## 4. Exact growth obstruction

### 4.1 Legal-zone update lemma [PROVEN]

**Lemma S1 (color-blind frontier update) [PROVEN].** Let `O` be a finite,
nonempty occupied set and let `x∉O`. Then

`L(O∪{x}) = (L(O)\{x}) ∪ (N_8({x})\(O∪{x}))`.                (S1)

In particular, the new term is available to whichever player moves next; the
owner of `x` is irrelevant.

*Proof.* By definition,

`N_8(O∪{x})=N_8(O)∪N_8({x})`.

Remove the newly occupied set `O∪{x}` and distribute the set difference. The
first term is `L(O)\{x}` and the second is the displayed new frontier. This is
also exactly the incremental operation in `LegalMoveStore::
update_for_placement_with_delta`: remove the placed coordinate, then insert
every empty radius-eight neighbor (`packages/hexo_engine/rust/src/legal.rs:
123-145`). ∎

**Corollary S1.1 (strict wrong-player expansion) [PROVEN].** Take

`O={(-8,0),(-16,0),(-24,0),(-32,0)}`, `x=(0,0)`, and `y=(8,0)`.

Then `x∈L(O)`, `y∉L(O)`, but `y∈L(O∪{x})`.

*Proof.* The exact distances are

`d(x,(-8,0))=8`, `min_{z∈O}d(y,z)=16`, and `d(y,x)=8`.

Also `x,y∉O` and `x≠y`. Apply the normal legality definition. ∎

This corollary is deliberately not labeled as an outcome counterexample. It
establishes strict enlargement of the opponent's action set, not that the
opponent can force a win from the enlarged set.

### 4.2 Reachable engine prefix witnessing the failed projection [PROVEN]

The strict expansion occurs on an actual legal, nonterminal engine history,
not only on an isolated set diagram. Let `F=Player0` and `S=Player1`, and play:

| placement index | owner | phase before | exact coordinate | earlier support at distance `≤8` |
|---:|---|---|---|---|
| 1 | `F` | `Opening` | `x=(0,0)` | compulsory opening |
| 2 | `S` | `FirstStone` | `(-8,0)` | `(0,0)`, distance `8` |
| 3 | `S` | `SecondStone` | `(-16,0)` | `(-8,0)`, distance `8` |
| 4 | `F` | `FirstStone` | `(-24,0)` | `(-16,0)`, distance `8` |
| 5 | `F` | `SecondStone` | `(-32,0)` | `(-24,0)`, distance `8` |
| 6 | `S` | `FirstStone` | `y=(8,0)` | `x=(0,0)`, distance `8` |

**Lemma S2 (history legality) [PROVEN].** Every placement in the table is legal
under the engine phase and growth rules, and no prefix is terminal.

*Proof.* The player/phase column follows the transition sequence proved in
§2.2. Every coordinate is new. The support column proves normal legality,
including the important fact that placement 3 may grow from placement 2 in the
same turn. Through placement 6, each player owns only three stones, so neither
can occupy a six-cell window. ∎

Immediately before placement 6 the occupied cells other than `x` are

`O^-={(-8,0),(-16,0),(-24,0),(-32,0)}`.

Their exact distances to `y=(8,0)` are, in the same order,

`16, 24, 32, 40`.

Thus `y∉L(O^-)`, while `y∈L(O^-∪{x})` because `d(y,x)=8`.

**Theorem S3 (normal legality is not deletion-monotone) [PROVEN].** On the
legal prefix through placement 5, erasing the compulsory `F` opening stone
`x=(0,0)` while retaining the normal-placement query sends the legal opponent
coordinate `S@(8,0)` to an illegal coordinate. Thus deletion of the natural
extra opening stone can remove a later opponent frontier move.

*Proof.* Lemma S2 proves that the real continuation is legal. The four exact
distances above prove that it is illegal after deletion. ∎

The board obtained by deletion is **not** asserted to be a legal full-game
history; §5 proves that the full shadow phases and stone counts already fail to
align. S3 isolates the additional board-level fact needed by any later
normal-placement coupling: even if opening alignment were repaired, simple
deletion would not preserve the opponent's legal frontier.

**Consequence for the classical proof [PROVEN].** A proof that relies on the
general rule “delete the unmatched own stone and copy every later opponent
move” cannot justify that rule in Hexo: a frontier-only move can fall outside
the domain of a strategy promised only on legal shadow histories. The usual
fallback for the different event “`σ` asks for the already occupied extra
cell” is irrelevant to this legality failure.

The conclusion is only about the identity/deletion coupling. A different
coupling might translate, delay, or otherwise encode frontier-only moves, but
it would need a new proof that it preserves turn order, legality, and
six-in-line completion. No such coupling is supplied here.

## 5. Independent opening-cadence obstruction

**Lemma S4 (the one-extra-stone shadow does not align at the opening) [PROVEN].**
After the compulsory opening and the second player's first completed turn, the
real placement counts are

`|X_F|=1`, `|X_S|=2`, with F to move at FirstStone.`

Deleting only F's opening stone changes real-label counts `(F,S)` from `(1,2)`
to `(0,2)`, hence the role-swapped shadow counts `(opener=S,second=F)` would be
`(2,0)`, not `(1,0)`. It cannot produce a legal
full-game prefix in which S has just acted as the opener, because every legal
full-game prefix immediately before the other player's first action contains
exactly one opener stone, at the compulsory opening coordinate.

*Proof.* The count and phase follow directly from

`F ; S,S ; F,F ; …`.

A legal full game begins with exactly one `Opening` placement before the other
player's `FirstStone`; this is enforced by `HexoState::new`, the origin-only
opening rule, and the `Opening→FirstStone` transition
(`packages/hexo_engine/rust/src/state.rs:149-160,317-323` and
`packages/hexo_engine/rust/src/rules.rs:16-23`). Deleting one F stone changes
the real-label tuple `(F,S)` from `(1,2)` to `(0,2)`, equivalently the shadow
role tuple `(opener=S,second=F)` is `(2,0)`, not the required `(1,0)`.
Translation can move a selected S stone to the origin but cannot change the
count two; the other S stone remains an unmatched extra opponent stone. ∎

This lemma is separate from the radius-eight failure. Even an unrestricted
placement variant with the same singleton-then-pairs cadence would need an
argument for absorbing that unmatched second-player stone. Conversely, even
if one first replaced the opening protocol by a cadence that aligned, Theorem
S3 would still refute the deletion projection under radius-eight growth.

## 6. Exact theorem status and missing replacement

### 6.1 Opener non-loss [OPEN]

**Target NL_F [OPEN].** “Engine `Player0` has a non-losing strategy from
`HexoState::new()`” is not proved or disproved here.

Theorems S3 and S4 rule out only the direct classical construction with one
ignored extra stone. They do not establish either of the stronger statements

- that adding an own stone can change a non-losing position into a losing one;
  or
- that engine `Player1` has a winning strategy.

Claiming either would inflate a transition-system obstruction into an outcome
theorem.

### 6.2 Minimal simulation and logic obligations [GAP]

A repaired stealing route must discharge the following obligations. The first
two concern one coherent simulated history; the third is the logical bridge
from that simulation to a non-losing strategy.

**GAP-OPENING-ALIGNMENT [GAP].** Starting from an arbitrary real prefix

`F@(0,0); S@a,S@b`,

construct a legal shadow prefix in which F can occupy the strategic role of
the assumed winning second player. The construction must account for both S
stones; it may not silently discard one as harmless.

**GAP-FRONTIER-COUPLING [GAP].** For every later real opponent placement made legal
only by a real extra stone, map that action to a legal shadow continuation and
map the shadow strategy's reply back to a legal real placement. The map must
preserve the ordered two-placement phase, including second placements that
grow from first placements, and it must preserve the implication “shadow six
for F gives a real six for F no later than an S win.” Identity after deleting
the extra stone fails by Theorem S3.

Proving these two obligations would repair the history-coupling part of the
specific route analyzed here; it would still be necessary to check the
ordinary already-occupied-prescription case and the determinacy bridge below.
Alternatively, a direct opener strategy could establish `NL_F` without
strategy stealing and without either shadow obligation.

**GAP-NONLOSS-DETERMINACY [GAP].** A contradiction to “S has a winning
strategy” is not syntactically the requested strategy
`exists sigma_F forall sigma_S, S never wins`. A completed stealing argument
must either construct `sigma_F` directly or invoke and instantiate the
appropriate determinacy theorem for the open payoff “S completes six at a
finite prefix,” under the declared infinite-game idealization. This bridge is
not proved in this bounded warm-up.

The sharp resume point is **GAP-FRONTIER-COUPLING**: find a non-identity
simulation invariant that admits moves such as `(8,0)` in §4.2, or prove an
outcome-level dominance theorem strong enough to tolerate every newly opened
opponent move. Static own-six monotonicity alone is insufficient.

## 7. Status ledger

| Claim | Status | Exact scope |
|---|---|---|
| Formal rule model in §2 | **PROVEN** | Production engine on the safe coordinate region, plus the explicitly declared unbounded-board idealization for infinite play |
| S1 legal-zone update | **PROVEN** | Every finite nonempty normal occupancy, under the declared unbounded-board rule |
| S1.1 strict new frontier | **PROVEN** | Exact sets `O`, `x=(0,0)`, `y=(8,0)` in §4.1 |
| S2 exact engine history | **PROVEN** | Six listed placements; source-level legality proof; no Cargo claim |
| S3 normal-legality deletion failure | **PROVEN** | Erasing the compulsory F opening stone `(0,0)` removes legal `S@(8,0)` |
| S4 singleton/pair cadence mismatch | **PROVEN** | Direct classical role-swap after the actual opening prefix |
| Unchanged deletion-shadow construction is invalid | **PROVEN** | S3 refutes growth closure; S4 refutes opening alignment |
| `NL_F`: opener has a non-losing strategy | **OPEN** | No opener or second-player outcome strategy supplied |
| GAP-OPENING-ALIGNMENT | **GAP** | Must absorb the second player's two-stone first turn |
| GAP-FRONTIER-COUPLING | **GAP** | Must handle opponent actions enabled only by real extra stones |
| GAP-NONLOSS-DETERMINACY | **GAP** | Must turn absence of an S winning strategy into one F non-losing strategy, or construct F directly |

There are no **VERIFIED** claims in this document: no finite enumeration or
machine test is used as evidence for a secondary-target claim. The
construction is checked by exact distance arithmetic against the audited
source predicates and is therefore labeled **PROVEN**, not **VERIFIED**.

## 8. Provenance

**Input state.** Branch `hunt/gap-raw`, HEAD
`283348dce09d42b67e364e0b2f2b63166b6b5f4d`. No commit was created.

**Required proof corpus read in order and in full.**

1. `GAP_RAW_PROOF_ROUND2.md`;
2. `GAP_RAW_REVIEW_ROUND2.md`;
3. `HUNT_REPORT_GAP_RAW.md`.

Those documents' blanket Maker–Breaker semantics were not imported into this
deliverable: the strategy-stealing target concerns the actual Maker–Maker
engine, in which either color's six is terminal.

**Rule sources read.** `packages/hexfield_eq/Cargo.toml`,
`packages/hexfield_eq/rust/src/lib.rs`,
`packages/hexfield_eq/rust/src/state.rs`, root `Cargo.toml`, and the imported
engine files `packages/hexo_engine/rust/src/{lib,coord,legal,rules,state,
tactics,board,error}.rs`. The engine package README and the independent
reference solver's no-draw comment at
`packages/hexfield_eq/rust/src/tss_reference.rs:158-163` were used only as
cross-checks; the formal rule statement above is tied to production engine
symbols.

**Machine work for this secondary target.** None. No secondary harness case was
added and no machine result is used here. The primary target's later Cargo run
and test-gated harness edit are recorded separately in
`GAP_RAW_PROOF_ROUND3.md`; no production source was changed.
