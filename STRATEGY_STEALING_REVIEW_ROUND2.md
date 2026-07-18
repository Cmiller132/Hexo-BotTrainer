# R-ST2-REV — Hostile review of `STRATEGY_STEALING_ROUND2.md`

**Reviewed artifact:** branch `hunt/gap-raw`, reviewed HEAD
`a85aa3116cb9eaff6a903233ee8b989d18030c82`.

**Required reading order:** `STRATEGY_STEALING_HEXO.md`, then the stealing
verdict in `GAP_RAW_REVIEW_ROUND3.md`, then `STRATEGY_STEALING_ROUND2.md`.

**Method boundary:** first-principles source and proof audit only. No Cargo
command, Lean build, harness, executable measurement, or production-source
edit was performed. The engine files audited were
`packages/hexo_engine/rust/src/{coord,legal,rules,state,tactics,board}.rs`.

**Overall verdict: CONFIRMED-WITH-ERRATA.** The ranked obstruction-class
result survives. S11 really excludes `C_iso`, S9.1 really excludes total
one-for-one shadows for every coordinate map at its stated no-invention
scope, and D2 really discharges the determinacy bridge. No outcome theorem is
proved. The document nevertheless needs several scope qualifiers and two
source-citation repairs before every surface-level `[PROVEN]` gloss is exact.

## Numbered findings

### 1. NOTE — the positive engine rule contract is confirmed citation by citation

> Radius eight is inclusive and color-blind
> (`packages/hexo_engine/rust/src/coord.rs:84-95`,
> `packages/hexo_engine/rust/src/legal.rs:17-18,123-145`, and
> `packages/hexo_engine/rust/src/board.rs:91-95,167-170`).

> The first placement of a pair updates the board and legal store before the
> second is validated (`board.rs:91-95`; `state.rs:293-335`).

> Along nonterminal play the ownership cadence is
> `F ; S,S ; F,F ; S,S ; F,F ; …`.

The cited source proves these facts exactly:

- `coord.rs:9-20` declares the `i16` carrier and origin, while `:76-82`
  implements `max(|dq|,|dr|,|dq+dr|)`.
- `coord.rs:84-95` uses inclusive ranges; `legal.rs:17-18,123-145` fixes
  radius eight and inserts every empty cell in the closed ball without an
  owner test; `board.rs:91-95,167-170` binds that update to occupancy only.
- `rules.rs:16-23` enforces the empty origin opening and `:34-44` enforces
  emptiness plus legal-store membership thereafter.
- `state.rs:149-160,317-335` gives the forced opener and the nonterminal
  cadence. `board.rs:91-95` mutates the board/legal store inside the first
  placement before the later second-placement call is validated at
  `state.rs:293-335`.
- `tactics.rs:13-17,21-75,205-208,451-485` gives length six, the three axes,
  the all-six predicate, and per-placement checking. `state.rs:302-337`
  checks the win before phase advancement, so a first-placement win suppresses
  the second placement.

The D6 derivation is also correct. The displayed `rho` and `kappa` permute the
three cube coordinates up to sign, hence preserve the maximum norm and the
unoriented axes; they fix the origin and transport the stored first coordinate
as claimed. Translations fail only at the rooted opening. The safe-carrier
qualification is honest: every displayed finite witness stays far from an
`i16` arithmetic boundary, while the global claims are assigned to the
`Z^2` idealization.

**Proposed repair:** none to the positive rule facts.

### 2. MINOR — two source ranges do not by themselves establish the full negative/local claim

> “`GameOutcome` has no draw alternative and the transition has no move-cap
> branch (`state.rs:64-71,289-337`).”

`state.rs:64-71` does prove that `GameOutcome` has only `winner` and
`placements`. The no-cap fact is also true, but `apply_with_delta` continues
through line 357; a range ending at line 337 cannot establish that no later
branch exists. Lines 339-357 construct and return the result without another
terminal condition.

> “This finiteness also follows directly from the finite-radius enumerator and
> stored legal-set iteration at `coord.rs:84-95`, `legal.rs:44-50,75-112`, and
> `state.rs:203-252`.”

Those ranges show a finite enumerator, the finite store, and its state-level
iteration, but the local citation omits both `LEGAL_RADIUS = 8` and the routine
that populates the store. The missing ranges were cited earlier in section
9.2, so this is not a factual defect.

**Proposed repair:** change the first citation to `state.rs:64-71,289-357`.
In D1, add `legal.rs:17-18,123-145` and, for the board binding,
`board.rs:167-170`.

### 3. MINOR — S7 is exact geometry, not an unconditional “S3 trap” criterion

> The exact **frontier-neutrality** condition is therefore not “the surplus is
> ours”; it is the geometric containment `B_8(x)⊆N_8(A)`.

> “Without it, deletion closure fails on every coordinate in the displayed
> difference.”

The set theorem is correct. Since `A⊆N_8(A)` and the premise `x∈L(A)` gives
`x∈N_8(A)`, every point of `B_8(x)\N_8(A)` is outside `A∪{x}`. Consequently

`L(A∪{x}) = L(A)\{x}`

exactly in the containment case, and otherwise

`L(A∪{x})\L(A) = B_8(x)\N_8(A) ≠ ∅`.

No geometric counterexample exists. There is, however, an engine-level
misclassification if “trap” means an actionable next-opponent move. Use the
legal S14 history immediately before its last placement. The occupied board
contains S at `(1,0),...,(5,0)` and F at
`(0,0),(0,8),(0,16),(0,24),(0,32)`. Let `x=(6,0)`. It is legal, strictly
q-extreme, and `y=(14,0)` lies in
`B_8(x)\N_8(A)`. S7 therefore detects geometric frontier growth, but
placing `x` completes S's six and terminates the engine. There is no next
opponent move at `y`. Thus noncontainment is not by itself equivalent to
strategic harmfulness.

**Proposed repair:** consistently call S7 “geometric frontier-neutrality.”
When interpreting case 2 as an S3-style reverse-projection obstruction, add
that the placement at `x` is nonwinning and its successor is nonterminal. If
`x` wins, the algebraic frontier difference is not exposed as an engine
continuation. S7 remains **CONFIRMED-WITH-ERRATA** rather than refuted.

### 4. NOTE — S5, S6, S7.1, S8, and S8.1 survive re-derivation

> If `A⊆B` are finite, nonempty occupied sets, `y∉B`, and `y∈L(A)`, then
> `y∈L(B)`.

> `{y∉B : y∈L(B) and y∉L(A)} = (N_8(E)\N_8(A))\B`. (S6)

The first statement is immediate from `N_8(A)⊆N_8(B)` plus real-board
emptiness. Expanding `N_8(B)=N_8(A)∪N_8(E)` proves the second exactly.
The exposed-surplus corollary is also exact: moving eight units farther in a
strict signed-cube extreme produces a point at distance at least nine from
every member of `A`.

S8 is ordinary set inclusion. S8.1's less obvious rigidity argument also
closes: consecutive source windows overlap in five points, so their injective
images must be consecutive intervals on one image line; reversal would make
windows two steps apart have equal images despite their four-point
intersection. Six consecutive windows isolate each lattice point, so
adjacency is preserved. The six distinct neighbors of a vertex then map onto
all six neighbors of its image, making the image neighbor-closed and hence,
by connectedness, surjective. The resulting triangular-grid automorphism is
exactly a translation followed by D6. Commutation with all origin-centered D6
maps leaves only `id` and the 180-degree rotation.

**Proposed repair:** none. The global hypothesis in S8.1 is essential and is
already disclosed; it does not exclude finite, dynamic, or proxy-assisted
encodings.

### 5. MINOR — S9 needs an explicit nonterminal actor-to-move premise

> A legal role-swapped shadow checkpoint at which F, now the shadow second
> player, is at `FirstStone` has shadow-role counts
>
> `(shadow opener S, shadow second F)=(2j+1,2j)` (11.2)
>
> for some `j≥0`.

The parity algebra is correct at a nonterminal checkpoint. At the real
F-checkpoint it equates

`(2k-b,2k-1-a)=(2j+1,2j)`,

and at the real S-checkpoint it equates

`(2k-b,2k+1-a)=(2j+1,2j+2)`.

Both give `a=b=2(k-j)-1`, a positive odd integer. This arithmetic is wholly
independent of the coordinate map.

The raw engine phase field alone does not imply the quoted counts, however.
A first-placement win bypasses phase advancement, leaving a terminal state
whose stored phase is still `FirstStone`. A concrete shadow-role history is

`S@(0,0); F@(1,0),F@(2,0); S@(0,8),S@(0,16);`

`F@(3,0),F@(4,0); S@(0,24),S@(0,32); F@(5,0),F@(0,1);`

`S@(0,40),S@(0,48); F@(6,0)`.

Every turn is legal. Before the last placement F has five consecutive
q-axis stones plus the off-line stone `(0,1)`; the last first placement
completes `{(1,0),...,(6,0)}`. The terminal engine state retains current
player F and stored phase `FirstStone`, but its shadow-role counts are `(7,7)`,
not `(2j+1,2j)`.

This does not rescue a simulation that must append another move: a terminal
shadow cannot satisfy that promise. It does make the unqualified count
identity false if “checkpoint” means merely the stored phase.

**Proposed repair:** write “legal nonterminal checkpoint at which the named
actor is to move at `FirstStone`” in the count identities, S9, S9.1, and the
relevant `C_iso` prose. Also replace “The smallest possible alignment” by
“The smallest count-compatible omission”; S9 proves necessity, not geometric
existence on every history. S9 is **CONFIRMED-WITH-ERRATA**.

### 6. MINOR — the headline drops the load-bearing no-invention qualifier

> The result is deliberately not an outcome theorem. It excludes two natural
> simulation families: every synchronous role-swapped shadow that maps all
> real stones one-for-one (irrespective of its coordinate map), and the second
> family that repairs the count mismatch by omitting stones but otherwise uses
> a no-invention, fixed-isometry, immediate-copy history map.

The formal corollary is narrower and correct:

> “No same-phase role-swapped shadow obtained by mapping **every** current
> real stone one-for-one, with no invented shadow stones, can be a legal
> `FirstStone` checkpoint for the corresponding actor.”

“With no invented shadow stones” matters. At a real F-checkpoint, total
role-swapped images have counts `(2k,2k-1)`; one invented opener-S stone and
one invented shadow-second-F stone give the count-compatible legal pattern
`(2k+1,2k)`. At a real S-checkpoint, `(2k,2k+1)` similarly becomes
`(2k+1,2k+2)`. This does not construct a valid proxy coupling, but it defeats
cadence arithmetic as an obstruction to the broader family literally stated
in the headline.

Later sections correctly list virtual/proxy stones as survivors, so the body
has not quietly proved the overbroad reading.

**Proposed repair:** change every summary to “every synchronous,
owner-faithful, **no-invention** role-swapped shadow mapping all real stones
injectively one-for-one, for every coordinate map.” Repeat the qualifier in
sections 12.4 and 15. S9.1 itself is **CONFIRMED** at its exact theorem scope;
the ranked summary is **CONFIRMED-WITH-ERRATA**.

### 7. NOTE — S10/S11 exhaust the fixed-isometry class, and the class is not vacuous

> “No member of `C_iso` exists. More locally: on the legal prefix of (12.1)
> through `F@z`, every no-invention, same-phase, role-swapped isometric shadow
> is forced to delete `F@x` and `S@u` and retain `S@v,F@w,F@z`; under every
> permitted `T`, the next legal real move `S@y` maps to an illegal shadow
> coordinate.”

At the witness checkpoint the available shadow-role counts are `(S,F)=(2,3)`.
A nonterminal shadow S-`FirstStone` checkpoint fitting inside those counts can
only have `(1,2)`, so it retains one of `{u,v}` and two of `{x,w,z}`.

- With opening `u`, only `x` is within eight and can be the F pair's first
  stone. From `{u,x}`, both `w` and `z` remain farther than eight.
- With opening `v`, only `w` can be first, after which `z` is legal at distance
  eight and `x` is not.

Thus `v;w,z` is the unique retained selection and legal order. It is a genuine
shadow prefix—for example, translation by `+(16,0)` roots `v` at the origin—so
items 1-3 are not definitionally inconsistent. But

`d(y,v)=24`, `d(y,w)=32`, and `d(y,z)=40`.

Every allowed `T=t∘g` preserves those distances and must root
the selected opening. Therefore `T(y)` is illegal, exactly as S11 claims. The
argument quantifies over every history-dependent choice of translation and D6
orientation at that checkpoint, not merely the translation used in the prior
review.

`C_iso` is a reasonable class for universal classical history projections,
but stronger than what every possible stealing proof needs: a
strategy-specific coupling need only cover histories generated by its own F
strategy against arbitrary S play. The document explicitly leaves that
survivor open. Dynamic recoding and proxies are also outside S8.1 because
S8.1 assumes one global window-exact injection.

**Proposed repair:** apart from the explicit nonterminal wording in Finding 5,
none. S10, S11, S11.1, and the survivor boundary are **CONFIRMED**.

### 8. NOTE — S12's preannouncement collision is exact at its stated scope

> “Then S has a legal response that destroys the promised exact copy: S plays
> `r` as its first coordinate.”

From `A⊆B`, `r∈L(A)`, and `r∉B`, S5 gives `r∈L(B)`. If
that move wins, the coupling has already lost. Otherwise `r` is S-occupied and
`rules.rs:34-44` bars the promised F copy. S can make a second placement in
the `Z^2` idealization: after the first placement, take a maximal-q occupied
cell and its empty outward `(1,0)` neighbor, which is legal at distance one.
If that second placement wins, the obstruction is only stronger.

The theorem does not claim to cover a second coordinate enabled only by its
own first, an already F-owned prescription, or a reply chosen after S acts.

**Proposed repair:** none. S12 is **CONFIRMED**.

### 9. NOTE — S13 defeats both FIFO orders and both choices of shadow opening

> No universal exact-copy implementation of the following one-sided schedule,
> using one translation/D6 isometry, can handle every legal history: the real
> opening `x` remains omitted; exactly one of the first real S pair is the
> shadow opening; the other is the sole FIFO-queued stone; the next real F pair
> is copied as the shadow F pair; and the queued S stone must be paired with S's
> next real placement. One checkpoint isometry `T` maps the selected opening,
> that F pair, and the queued/new S pair; no proxy or other represented support
> is allowed.

The witness distances recheck as

`d(a,x)=8`, `d(b,a)=8`, `d(p,a)=4`, `d(p,b)=8`,
`d(q,p)=8`, and `d(c,x)=8`.

Thus the real history is legal and nonterminal. For either chosen opening,
`p,q` is a legal shadow-F pair and the queued other member of `{a,b}` is legal
at distance eight. Yet

`d(c,a),d(c,b),d(c,p),d(c,q)=16,24,20,20`.

If `c` is first, it is unsupported; if the queued stone is first, it still
does not support `c`. Isometries preserve the four failures. Reversing `p,q`
does not add an omitted case because `q` is not legal first from either
opening.

**Proposed repair:** none. S13 is **CONFIRMED** only for the literal one-queue,
one-isometry, no-proxy schedule it states.

### 10. NOTE — S14's terminal-lag count obstruction is exact

> “At that terminal checkpoint a literal one-S-stone-lag shadow represents at
> most five S stones. It also has at most the five literal F stones. With no
> proxy, duplication, or recoloring, neither shadow color can own a
> six-window.”

In the displayed history, S grows consecutively through `(1,0),...,(6,0)`;
F grows legally along the distance-eight chain
`(0,0),(0,8),(0,16),(0,24),(0,32)`. S has only five aligned stones before the
last move, and F has five total. The last S placement is therefore the first
terminal placement and completes exactly the displayed q-axis six. A literal
one-S-stone lag leaves at most five images of either owner, so it cannot have
terminated earlier or at that checkpoint.

**Proposed repair:** none. S14 is **CONFIRMED** at its universal,
owner-faithful, no-proxy/no-duplication scope. A strategy-generated invariant
that excludes this history remains outside the theorem, as the source says.

### 11. MINOR — D1's macro-game is valid, but macro-to-sequential expansion omits off-path totality

> “Therefore a pure single-placement strategy induces exactly one macro
> choice—singleton if `c_1` wins, otherwise its ordered pair—and every legal
> macro strategy expands back into the corresponding two sequential
> prescriptions.”

The game-tree construction itself is sound. One fully explicit version uses
the countable tagged alphabet

`Z^2 ⊔ (Z^2 × Z^2) ⊔ {dummy}`,

where the first tag is a winning singleton, the second a legal ordered pair
with nonwinning first placement, and `dummy` is the sole successor after a
terminal prefix. Legal macro-prefixes and their dummy extensions form a
prefix-closed, pruned tree:

- a finite occupancy with `n≥1` has at most `216n` legal first coordinates,
  because a closed radius-eight ball has 217 cells;
- after a nonwinning first placement, the second-coordinate set is finite;
- a maximal-q occupied cell has an empty outward neighbor at distance one in
  `Z^2`, both at a macro boundary and after a nonwinning first placement; and
- every terminal node has its unique dummy continuation.

The S-win payoff is the union of the cylinders based at finite S-terminal
prefixes, hence is open in the relative product topology on this branch
space. Finitely branching is stronger than the cited theorem needs, but it is
genuinely proved here.

The quoted strategy equivalence leaves one formal hole. A standard pure
single-placement strategy is total on all of its decision nodes. Expanding a
macro strategy supplies its `c_2` at the `SecondStone` node reached after its
own prescribed `c_1`, but says nothing at an off-strategy `SecondStone` node
whose observed first coordinate differs. Such nodes do not affect plays
consistent with the strategy, so the winning-strategy correspondence is not
damaged.

**Proposed repair:** fix an enumeration of `Z^2`; at an inconsistent
`SecondStone` history prescribe the least legal coordinate. The non-dead-end
argument guarantees one. D1 is **CONFIRMED-WITH-ERRATA**.

### 12. NOTE — the Gale–Stewart citation and D2 equivalence are exact

> **Theorem D2 (non-loss bridge) [PROVEN from the CITED theorem].** In the
> unbounded Hexo idealization,
> `NL_F ⇔ S has no winning strategy`.

[Martin's cited text](https://www.math.ucla.edu/~dam/booketc/D.A._Martin%2C_Determinacy_of_Infinitely_Long_Games.pdf)
defines the branch topology by finite-prefix cylinders in section 1.1 and at
Theorem 1.2.4, printed page 15, states that all open games are determined (and
attributes the result to Gale and Stewart). Section 1.2 begins at printed page
12. The cited original chapter title, authors, 1953 date, pages 245-266, and
[DOI](https://doi.org/10.1515/9781400881970-014) are also correct.

Apply the theorem with S as the first macro-player and payoff `W_S`. Either S
has a pure strategy forcing a finite S-winning prefix, or F has a pure
strategy forcing the complement. That complement consists exactly of an
underlying finite F win (followed only by dummy symbols) or a genuinely
infinite nonterminal history with neither color completing six. The latter is
precisely round 1's declared meta-level draw. Immediate engine termination
precludes a branch with an earlier win by both players.

Thus F's complement-winning strategy is literally
`exists sigma_F, forall sigma_S, S never wins`. Mutual exclusivity follows by
playing two purported winning strategies against one another. D2 does not
select which determinacy alternative holds and therefore does not prove
`NL_F`.

**Proposed repair:** none. The cited theorem and D2 are **CONFIRMED**.

### 13. NOTE — status discipline, S3/S4 scope, and the survivor list are honest

> “Dynamic recoding, virtual bookkeeping stones, and strategy-specific
> invariants that do not promise a map on every legal history survive.”

> “This is an exact boundary for `C_iso`, not a theorem against all
> non-identity simulations and not an outcome result.”

Round 2 does not resurrect the refuted identity coupling. It uses S3's same
frontier-only move inside newly defined fixed-isometry and FIFO classes, and
it supplies new exhaustive arguments for those classes. S4 remains only the
direct one-deletion cadence mismatch; S9 replaces it with count arithmetic
under explicit owner-faithful/no-invention assumptions.

The survivors are genuinely outside the proved exclusions:

- proxy or virtual stones violate no-invention;
- dynamic recoding violates the one-checkpoint isometry/immediate-copy rule;
- more complex lag violates S12/S13/S14's exact causal or queue premises; and
- a strategy-specific invariant need not map witness histories that its own
  F strategy cannot generate.

Nothing in S8.1 closes these routes because its injection is global and
window-exact. Nothing in D2 refutes S's winning-strategy alternative. The
ledger correctly leaves `NL_F`, the broader opening alignment, and the
broader frontier coupling open.

**Proposed repair:** add the no-invention and nonterminal qualifiers from
Findings 5-6 to the summaries; otherwise no status change. The survivor
boundary and no-outcome-inflation discipline are **CONFIRMED**.

### 14. NOTE — provenance is accurate as an input statement but should name the reviewed artifact

> “**Input state.** Branch `hunt/gap-raw`, HEAD `12980bc8`. This authoring pass
> created no commit.”

Commit `12980bc8` is indeed the parent/input to the round-2 artifact. The
document reviewed here is committed at `a85aa311`. Those facts are compatible
with a no-commit authoring pass, but recording only the input hash makes the
artifact identity unnecessarily ambiguous.

**Proposed repair:** retain `12980bc8` as the input/base and add
`a85aa3116cb9eaff6a903233ee8b989d18030c82` as the reviewed/output artifact.

## Per-result verdicts

| Result | Source status | Review verdict | Exact disposition |
|---|---|---|---|
| Inherited rule contract | PROVEN | **CONFIRMED-WITH-ERRATA** | All facts match production; extend the no-cap and local finite-store citations (Findings 1-2) |
| Rule-level D6 equivariance | PROVEN | **CONFIRMED** | Correct on `Z^2` and on safely transformed executable histories |
| S5 reply lifting | PROVEN | **CONFIRMED** | Exact subset-plus-emptiness implication |
| S6 frontier-gap formula | PROVEN | **CONFIRMED** | Exact set difference for coordinates empty in the real board |
| S7 one-surplus dichotomy | PROVEN | **CONFIRMED-WITH-ERRATA** | Exact geometric `L`-set criterion; a nonwinning/nonterminal guard is needed for an actionable S3 trap |
| S7.1 exposed surplus | PROVEN | **CONFIRMED** | Strict signed-cube extremality produces a geometric gap |
| S8 same-color win monotonicity | PROVEN | **CONFIRMED** | Superset retains the same six-window |
| S8.1 global window-exact rigidity and consequence | PROVEN | **CONFIRMED** | Injection is forced to be translation-D6; finite/dynamic/proxy maps remain outside it |
| S9 equal-odd deletion law | PROVEN | **CONFIRMED-WITH-ERRATA** | Algebra exact for a nonterminal actor-to-move checkpoint; raw terminal `FirstStone` storage is an omitted case |
| S9.1 total one-for-one obstruction | PROVEN | **CONFIRMED** | Covers every coordinate map, not merely isometries, under owner fidelity and no invention |
| Opening/headline summary of S9.1 | PROVEN prose | **CONFIRMED-WITH-ERRATA** | Must retain the no-invention qualifier |
| S10 witness replay | PROVEN | **CONFIRMED** | All five normal supports are at distance eight; no displayed prefix is terminal |
| S11 `C_iso` obstruction | PROVEN | **CONFIRMED** | Selection/order and every permitted history-dependent checkpoint isometry are exhausted |
| S11.1 D6 witness orbit | PROVEN | **CONFIRMED** | The proof uses only D6-invariant distances, owners, and counts |
| S12 preannounced collision | PROVEN | **CONFIRMED** | Exact only for a still-empty first coordinate fixed before S acts |
| S13 FIFO obstruction | PROVEN | **CONFIRMED** | Both opening choices and both queue/new orders fail on the witness |
| S14 terminal-lag obstruction | PROVEN | **CONFIRMED** | Literal one-S-stone lag has fewer than six images of either owner |
| Gale–Stewart open determinacy citation | CITED | **CONFIRMED** | Martin Theorem 1.2.4 and the original bibliographic citation check out |
| D1 Hexo S-win is open | PROVEN | **CONFIRMED-WITH-ERRATA** | Tree, pruning, finite branching, and openness are correct; fill off-path sequential prescriptions |
| D2 `NL_F ⇔` no S winning strategy | PROVEN from CITED | **CONFIRMED** | Pure open-game determinacy gives exactly the complement strategy |
| `GAP-NONLOSS-DETERMINACY` discharged | PROVEN from CITED | **CONFIRMED** | No additional constructive F strategy is logically required after refuting an S winning strategy |
| Ranked outcome (b), S9+S11 | PROVEN | **CONFIRMED-WITH-ERRATA** | Genuine obstruction class; summary qualifiers must match the theorem premises |
| `NL_F` | OPEN | **CONFIRMED** | Neither determinacy alternative is selected |
| Broader opening/frontier coupling and proxy-shadow routes | OPEN | **CONFIRMED** | Dynamic, proxy, lagged, and strategy-specific mechanisms remain unexcluded |

No result is **REFUTED** or **DOWNGRADE**. The corrections narrow prose to the
already-proved hypotheses; they do not reduce a proved theorem to a sketch or
an open claim.

## Overall disposition

**CONFIRMED-WITH-ERRATA.** The most severe issue is Finding 6: without the
no-invention premise, the advertised “every coordinate map” cadence exclusion
is false, because one proxy of each shadow color repairs the raw count pattern.
The formal S9.1 theorem contains that premise, S11 is exhaustive for `C_iso`,
and the survivor list admits proxies, so the ranked obstruction-class theorem
survives. Findings 3, 5, and 11 require local terminal/totality scope repairs;
Finding 2 requires citation repairs. There is no FATAL or MAJOR defect and no
outcome-theorem inflation.
