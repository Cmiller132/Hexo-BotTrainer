# R-KT1 forced-reply kernel taxonomy

Date: 2026-07-17  
Branch/base: `hunt/kernel-taxonomy` / `369e6969`  
Round: hunt + shadow only; no Lean and no consumption wiring

## 1. Scope and source ruling

This taxonomy is for defender `Universal`/AND contexts. A reply kernel is a
set of defender continuations whose successful coverage would justify omitting
the complement. The tested load-bearing condition is therefore stronger than
the historical G2R7 OR observation: an out-of-kernel defense must be resolved
as a defender refutation while every in-kernel defense is resolved as a
claimant proof before it is a counterexample. `Unknown` is never treated as a
refutation.

The binding local grammar is `docs/PROOF_TSS_DEFENDER_ZONES.md` as requested.
It contains D1--D13, T3--T8, and P1--P3. It does **not** contain the D19--D21
or L15 text cited by `RESEARCH_AGENDA.md` and
`docs/PLAN_TSS_SOLVER_UPGRADES.md`. The later protected-gate definitions can be
found in `E:\tss-lean\SOURCE_PROOF.md`, but that file was not named as the
normative grammar for this round. Consequently, D19 checkpoint identity,
`Q_N^D`, and `GateAdaptiveEscape` are recorded below as unavailable
certificate metadata, not reverse-engineered from a bare position.

There is also an implementation seam correction. The proven Q8 `K_reply`
hook in `BUILD_K_REPLY_CONSUME.md` filters a claimant `Choice` fallback at an
urgent claimant `SecondStone` position; it does not run at a defender AND
node and it is not routed through the official wide PN search. The official
wide engine already applies T6 at forced AND nodes:

- at defender `SecondStone`, the enumerated T6 reply set has size at most 2;
- at defender `FirstStone`, the T6 cell kernel has size at most 4 and the
  engine normally emits complete unordered `DefenderPair` children.

Thus the live post-T6 seam is not a presumed `<=37`-wide AND seam. The shadow
records full legal width, T6 reply-cell width, actual child width, and proposed
kernel width separately. A future `<=37 -> <=k` claim must target a pre-T6 or
unforced-zone seam and cannot be inferred from post-T6 child counts.

## 2. Decidable axes

| Axis requested by the agenda | Position/engine support | Taxonomy use |
|---|---|---|
| Urgency | Yes. The maintained `WindowStore` exposes claimant live count-4 and count-5 windows, `own_win_now`, and `tau`. | `C4`, `C5`, `MIXED`; `EMPTY` is retained as an invariant-check bucket. |
| Escape phase | Yes for the game phase: defender `FirstStone` has budget 2; defender `SecondStone` has budget 1. | `F2` and `S1`. |
| Gate adjacency | Yes only as the exact incidence graph between T6 reply cells and complete commuting defender pairs. | Pair graph `G_Q=(V,E)` and its cover number. |
| Protected D19 gate/checkpoint adjacency | No. A bare position has no named family, copied-child map, checkpoint roles, or protected-gate tag. | `NO-CONJECTURE` this round. |
| Touched versus virgin current windows | The current board and `WindowStore` can distinguish touched and all-empty windows. At a forced T6 node, however, every kernel cell belongs to a claimant-touched live threat, so this axis is constant. | No forced-node split. |
| D21 dangerous exposure (`Q_N^D`) and L15 escape state | No. These are certificate-recursive clocks/modes, not current-position fields. | `NO-CONJECTURE` this round. |
| P1/P2 local domination | Conservatively decidable from the 18 incident window entries plus the radius-8 legal-frontier support predicate. | `S1_DEAD_SPOKE_C4`. |
| P3 same-turn commutation | Yes. The existing pair plan checks both directed orders and exact final-position identity. | `F2_COVERk_*` projection diagnostic. |

No axis below depends on a history fact absent from `HexoState`, on a scan of
future certificates, or on an invented gate tag.

## 3. Class grid

The urgency suffixes are crossed with every applicable row:

| Suffix | Current claimant threat family |
|---|---|
| `C4` | live count-4 windows only |
| `C5` | live count-5 windows only |
| `MIXED` | both count-4 and count-5 windows |
| `EMPTY` | neither; an invariant/unsupported bucket at a forced AND node |

The substantive rows are:

| Class family | Membership | Proposed `K` | Target | Phase-B status |
|---|---|---|---:|---|
| `S1_SINGLETON_{C4,C5,MIXED,EMPTY}` | Defender `SecondStone`; current exact T6 set is `{x}`. | `{x}`. | 1 | Proven baseline; shadow measures coverage, not a new post-T6 cut. |
| `S1_DEAD_SPOKE_C4` | Defender `SecondStone`; T6 set `{x,y}`; `x,y` are the two empties of a claimant count-4 witness; every other incident window is dead; adding either cell gives the identical radius-8 legal-frontier support. | The deterministic representative `min(q,r){x,y}`. | 1 | New narrower conjecture; shadowed. |
| `S1_NO_CONJECTURE_{C4,C5,MIXED,EMPTY}` | Any remaining defender `SecondStone` node, normally a generic two-cell T6 set. | None. The two cells may affect different live windows or legal-frontier support. | -- | `NO-CONJECTURE`; measured but skipped. |
| `F2_COVERk_{C4,C5,MIXED,EMPTY}`, `k=1..4` | Defender `FirstStone`; every child is an exact commuting unordered pair; the complete-pair graph has a minimum vertex cover `C` of size `k`. | First-placement projection `C`. After `c in C`, retain every legal second endpoint incident to `c`. | `k<=4` first placements | Shadowed as a P3 projection. **Not an AND-child kernel:** every unordered-pair obligation remains, so it is ineligible for the proof queue. |
| `F2_NO_SMALL_COVER_{C4,C5,MIXED,EMPTY}` | Same phase, but no cover of size at most 4 is found. | None. | -- | `NO-CONJECTURE`; invariant/extension bucket. |
| `F2_UNCOMPRESSED_{C4,C5,MIXED,EMPTY}` | Defender `FirstStone` represented as one-placement children rather than validated complete pairs. | None. P3 does not license dropping a first placement until its complete-turn continuations and reverse keys are known. | -- | `NO-CONJECTURE`; measured but skipped. |
| `AND_UNSUPPORTED_{EMPTY,OPENING,MOVE_SHAPE}` | Empty, opening-phase, or mixed/unknown Universal representation. | None. | -- | `NO-CONJECTURE`; counted separately from traversal corruption. |
| `D19_GATE_*` | Named protected exact-copy gate with checkpoint roles and L15 escape contract. | The external, non-normative text defines `K={d : tau(F_Q \ d) <= b-1}`. | finite; at most 4 for the current size-one/two threat sets | `NO-CONJECTURE` in this worktree because membership is not position-decidable. |
| `D21_{TOUCHED,VIRGIN,DANGEROUS}_*` | Ordinary debited-zone context with recursive `Q_N^D` exposure. | No position-only rule. | -- | `NO-CONJECTURE`; certificate metadata unavailable. |

The class names are deliberately explicit in the raw log. Empty or impossible
cross-products remain visible instead of silently being merged into a more
favorable class.

## 4. Kernel rationales and incremental triggers

### `S1_SINGLETON_*` -- exact T6 baseline

Let `F` be the claimant's live threat-empty family at a forced budget-one
defender node. T6 retains the intersection `K1 = intersection(F)`. If
`K1={x}`, any legal reply outside `x` leaves at least one live threat unhit and
ends the defender's turn; L1/T6 supplies the claimant completion. This is the
same forced-reply geometry underlying the already-proven urgent Q8 class,
although the production hooks are on opposite node kinds. It is not a new
post-T6 reduction because the wide solver already enumerates only `x`.

An O(1)-per-move trigger can maintain, for each live count-4/5 window, its
empty mask and maintain a two-cell-or-overflow intersection summary for the
active claimant family. Make/unmake touches at most 18 length-six windows and
updates the summary plus `tau`; the phase and claimant identity are already in
the position key. No board rescan is needed.

### `S1_DEAD_SPOKE_C4` -- P2 representative

Here T6 leaves `{x,y}` and the exact P2 premises say the cells are
interchangeable: their only live tactical role is the same count-4 witness,
all other incident windows are already two-coloured, and either placement
induces exactly the same legal-frontier support. Replacing defense `y` with
`x` therefore still hits the witness and neither opens nor removes a later
legal continuation. P2 supplies the domination relation; choosing one
deterministic representative proposes a genuine `2 -> 1` current-AND cut.
The predicate is intentionally strict—nearby or visually symmetric cells are
not enough.

For an incremental trigger, maintain per-cell counts of non-dead incident
windows and the identity of the sole count-4 witness. Maintain a reference
count for radius-8 legal support for every current frontier cell; make/unmake
changes only the radius-8 ball of the placed cell. Two 434-bit support
signatures (or their reference-count-backed hashes plus exact comparison on a
fire) decide frontier equality. Phase, `tau=1`, and the two-cell T6 summary are
already constant-size inputs. The shadow implementation recomputes this exact
predicate for isolation; production wiring would use these maintained fields.

### `S1_NO_CONJECTURE_*`

For a generic two-cell intersection there is no honest domination argument.
One cell can participate in a different live window, kill a different future
window, or change the radius-8 legal frontier. P1 only dismisses a dead cell,
and a member of the exact T6 intersection is not dead merely because its mate
also hits the current family. Selecting one representative without the full
P2 premises would simply assume the theorem being sought, so Phase B records
the class's economic mass but does not test a fabricated kernel.

The class itself is incrementally decidable by the same maintained T6
intersection summary. Any future refinement must add a position-decidable
local signature and a real replacement lemma.

### `F2_COVERk_*` -- P3 projection, not an AND-width cut

Let vertices be exact T6 first placements and let an edge `{x,y}` denote a
validated complete defender turn. The existing pair plan proves both orders
legal and their final keys identical. If `C` is a vertex cover, every turn
started outside `C` has its other endpoint in `C`; commute the placements and
the identical final state is reached from a retained first placement. This is
a sound domination argument for first-placement dispatch.

It does **not** dominate one complete unordered pair by another. After
projection the solver must still discharge every edge of `E`, and the current
wide PN node already stores exactly those unordered edges. The shadow therefore
tests the projected load-bearing statuses but the class is not a surviving
AND-child kernel and has zero current child-work saving.

An incremental form maintains the at-most-four T6 vertices and a six-bit
unordered edge mask. Each make/unmake updates the affected threat-empty masks;
the existing reverse-key check supplies edge admission. A fixed-parameter
branch on an uncovered edge finds a cover of size at most four in at most
`2^4` branches, a constant independent of board size. A canonical-frame order
would select among multiple covers in future production code.

### `F2_NO_SMALL_COVER_*` and `F2_UNCOMPRESSED_*`

There is no conjecture. In the former, a small first-placement projection was
not found. In the latter, exact complete-turn adjacency and reverse-position
identity are absent, so P3 cannot be invoked. Both classes are important
negative taxonomy data: they prevent an implementation detail from being
mistaken for a certified reply kernel.

The no-small-cover bucket uses the same constant edge mask. The uncompressed
bucket is decided directly from the node's child representation and phase.

### Protected gates and touched/virgin escape classes

The current position can maintain window counts, touched bits, and current
distances incrementally, but it cannot reconstruct a D19 named family, copied
child map, checkpoint role, recursive `Q_N^D`, or whether a certificate path
has entered L15's adaptive escape. Any kernel on those axes would require the
certificate builder/verifier to carry those fields explicitly. Until that
state exists in both producer and strict verifier, these rows are
`NO-CONJECTURE`, not zero-fire successes.

## 5. Finite width facts at the measured seam

For Connect-6 threat windows every live threat-empty set has size one or two.
At `tau=1`, the exact intersection therefore has at most two cells. At
`tau=2`, choose one minimum cover `{a,b}`. A size-two threat not containing
`a` forces its other cover endpoint through `b`, and symmetrically for `b`;
the family admits at most the base cover and one alternative on each side,
with at most one off-both combination. Hence the union of vertices that occur
in a minimum two-cover has size at most four, and the unordered complete-pair
width is at most six. Two disjoint two-cell threats make the four-vertex bound
sharp.

This algebra explains why no post-T6 `37 -> k` event can occur in the current
wide engine. It does not prove that a pre-T6 legal or unforced-zone kernel is
impossible; it identifies the seam that a future round would have to expose.

## 6. Promotion rule

A class enters the ordered Lean queue only if all of the following hold:

1. it is a complete defender-reply kernel rather than a projection;
2. every measured load-bearing case has zero counterexamples, with unresolved
   cases reported separately;
3. it removes a material share of actual defender child work, not merely full
   legal cells already dismissed by T6;
4. its trigger can be maintained on make/unmake without a position rescan;
5. all premises are independently re-derived by the strict verifier.

The measured verdict and resulting queue are in
`HUNT_REPORT_KERNEL_TAXONOMY.md`.
