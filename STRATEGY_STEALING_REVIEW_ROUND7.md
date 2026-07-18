# R-ST7-REV — Hostile review of `STRATEGY_STEALING_ROUND7.md`

## Method and review boundary

**Reviewed artifact.** `STRATEGY_STEALING_ROUND7.md` first appears at
landing commit `798bfb75cee282960640fd8b4abcbfea47c11404` on branch
`hunt/gap-raw`. Its Git blob is
`2241724248da0bb66b01d7d1764496b1562f0fb5` and its SHA-256 is
`12e228fe0d8aba68fe58ff090960d522627116ac194644ce3af58df4e8796140`.
The worktree copy is byte-identical to that first landed blob. The branch had
advanced to `c57da44286f75feb236e6da6c55cdd53e5ec2e68` when this review
was performed, but the reviewed file remained unchanged.

**Named input.** The artifact names input
`09e27a937c7ae6be4f0c7d32b02d2bcd3d885621`. Git confirms that input is
an ancestor of the landing. The immediate landing parent is
`a8a0b92d641b690b63d43f049d2b4c2fa0d4e9c1`; the landing adds the round-7
artifact and its authoring prompt, and the artifact is absent from that
parent. A name-only lineage check shows that the pre-existing predecessor
strategy-stealing corpus and the six engine rule files did not change from
the named input through the intermediate parent or landing; the landing adds
only round 7 and its authoring prompt. Git cannot preserve an uncommitted
authoring buffer, so “landed unmodified” is confirmed to the observable
extent: the current file exactly matches the first committed artifact at
`798bfb75`.

**Required reading completed first, in order and in full:**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md` and
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md` and
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including binding section 35, and
   `STRATEGY_STEALING_REVIEW_ROUND4.md`;
5. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, and
   `STRATEGY_STEALING_REVIEW_ROUND5.md`;
6. `STRATEGY_STEALING_ROUND6.md`, including binding section 53, and
   `STRATEGY_STEALING_REVIEW_ROUND6.md`; and
7. `STRATEGY_STEALING_ROUND7.md`.

I then read in full and checked the cited predicates in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
This was a first-principles proof audit. I ran no Cargo command, Lean build,
harness, executable search, or proof-search program. Every coordinate,
distance, cadence, occupancy count, six-window census, deficit, debt update,
terminal age, and transversal below was recomputed by hand. I did not open or
use a `GAP_RAW_*` proof or review as mathematical evidence. Git names and
tree metadata were used only for provenance. Pre-existing unrelated untracked
entries were left untouched. No commit was created; the only deliverable
written by this review is `STRATEGY_STEALING_REVIEW_ROUND7.md`.

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** No theorem is refuted and no
MAJOR defect was found. S46–S54 survive at their stated conditional or finite
scopes. The source needs five small formal or documentary repairs: totalize
S41's pathwise portfolio policy explicitly; state S50's common-axis inference;
call S51 a non-disjoint stop cover; restore two caveats omitted from the
authoritative ledgers; and record the actual landing commit. None supplies
universal alignment, a global
`P0–P6/P5R` coupling, the full per-pair `K=2` carrier, or `NL_F`.

## Numbered findings

The findings lead with the concrete witness and count stress tests, then audit
the abstract debt/lock machinery; the verdict table below restores source
order.

### 1. NOTE — the production rule and terminal-event contracts are exact

> “A normal coordinate is legal exactly when it is physically empty and
> belongs to the color-blind radius-eight legal store.”
>
> “Section 53's terminal-closure definition supersedes round-5 clauses 5--6
> on the final paired F microstep.”

**Independent recomputation.** `coord.rs:76-95` implements axial
maximum-norm distance and the inclusive ball. `legal.rs:17-18,123-145`
uses radius eight, removes occupied cells, and adds empty halo cells without
an owner test. `rules.rs:11-44` rejects play after terminality, fixes the
rooted origin opening, rejects occupancy, and checks the maintained legal
store at both ordinary phases. `board.rs:83-105` inserts the physical
stone before updating legality and windows. `state.rs:203-252,265-273,
289-357` gives terminal no-continuation, append-only forward history,
immediate win testing after each append, and phase advance only after a
nonwin. `tactics.rs:13-17,21-75,205-208,451-485` gives the three axes,
six-cell windows, eighteen incident windows per coordinate, and the physical
all-six test.

Accordingly, within one engine history a winning first placement suppresses
its second placement. In the artificial paired carrier, round-6 section 53
separately makes the final cross-board microstep atomic: both legal physical
appends are installed before that coupled event closes. Round 7 uses that
precedence consistently in Definitions 55.1 and 55.3, S47, and S52's four-row
event table. No undo, recoloring, or label-only terminal certificate appears.

**Proposed repair:** none.

### 2. NOTE — S48's S41 repayment and S42 red line recompute exactly

> “The S41 terminal trace admits `F-CAD_2^st` and its canonical events
> satisfy `F-LOCK`, while the S42 trace violates terminal readiness before
> its final prescription.”
>
> “The existing canonical service rule therefore does **not** maintain the
> state invariant on every `A_FS2` trace.”

**Independent recomputation: S41.** Let

- `V={(1,0),...,(6,0)}`,
- `V_2={(2,0),...,(7,0)}`, and
- `W={(0,0),...,(5,0)}`.

The relevant ledger is:

| Reached stage/event | `delta_H(V)` | `delta_R(W)` | Debt |
|---|---:|---:|---:|
| first pre-query after the seed service and intervening rolling S pair | 2 | 3 | 1 |
| service 1, both events off `V/W` | 2 | 3 | 1 |
| after `z=(-8,0), k=(3,0)` | 2 | 2 | 0 |
| after `z=(5,0), k=(4,0)` | 1 | 1 | 0 |
| final pre-query: `V` / `V_2` | 1 / 2 | 1 / 1 | 0 / -1 |
| terminal `z=(6,0), k=(5,0)` (arithmetic only; no later portfolio) | 0 | 0 | 0 |

The first event of service 2 is real-only on the selected windows and repays
exactly one debt. Its second event hits both and preserves zero. The
intervening rolling pair blocks neither line. Before `z=(5,0)`, the only
unblocked shadow q-window with deficit at most two is `V`: q-windows
through `(0,0)` are `Shat`-blocked and the off-line `Fhat` runs
have length at most two. After that append `V_2` is the only additional
near window, and both can use `W`. The final canonical cell is the unique
real hole, so LOCK holds.

**Independent recomputation: S42.** Before its second post-S15 service pair,
the shadow q-window has deficit two and the displayed real q-window deficit
three. The first event `z=(5,0), k=(3,0)` hits both, leaving shadow
deficit one and real deficit two. More strongly, real F then has only four
stones in total, so *every* real six-window has deficit at least two. No
portfolio—not merely the displayed assignment—can satisfy terminal readiness
before `z=(6,0)`. That prescription is the shadow's sixth stone and wins
only there.

S42 supplies a legal live `A_FS2` handler prefix through the first event,
passing first-safety, certificate validity, and `tau_E=0`, followed by
its raw shadow-terminal extension. Readiness already fails at the live
pre-query state before `z=(6,0)`. It therefore refutes universal
preservation through every live canonical handler extension from
`A_FS2`, which is the precise content of the artifact's negation. It does
not claim that every canonical prefix fails or that every possible selector
fails.

**Proposed repair:** none beyond Finding 9's policy-totalization sentence.

### 3. NOTE — S49's sixth-stone barrier is selector-independent at exactly its stated scope

> “Fix ... any genuine one-for-one F-role event continuation with no unmatched
> real-F placement.”
>
> “This remains true for every choice, ordering, or augmentation of the two
> existing real service placements.”

**Independent recomputation.** S15 starts with
`(|X_Fhat|,|X_F|)=(2,1)`. One-for-one events give:

| Post-S15 event | `Fhat` stones | real-F stones | Cadence location |
|---:|---:|---:|---|
| 1 | 3 | 2 | first post-S15 F turn, first |
| 2 | 4 | 3 | first post-S15 F turn, second |
| 3 | 5 | 4 | second post-S15 F turn, first |
| 4 | 6 | 5 | second post-S15 F turn, second |

The first three shadow appends cannot win by count. If event four is
shadow-terminal, its paired real append leaves only five real-F stones, so a
real six is impossible. Replacing, reordering, or augmenting the choice rule
for the two legal service cells cannot alter these counts, and the engine
cadence supplies no third real-F placement in that turn.

This is exhaustive for the formal qualifier: post-S15 one-for-one events,
with no unmatched real-F append. It does not address a different legal
carrier that somehow earns extra real-F placements at another phase. The
age-six condition is substantive, not vacuous: S42 realizes it on a legal
strategy-own history. What remains unproved is that a globally alleged-winning
`sigma` has such a history.

S49.1 also has the right quantifier. Any property promised at every S15 base,
preserved by every full live canonical extension for every legal pure
strategy, and implying first-terminal alignment is contradicted by S42. A
property restricted to alleged-winning histories or to CAD admission is not
excluded.

**Proposed repair:** whenever “every augmentation” is quoted out of context,
repeat the one-for-one/no-unmatched-real-placement qualifier.

### 4. MINOR — S50's construction is sound, but the common-axis step should be explicit

> “Any urgent window for this final debt must contain at least four real-S
> stones. There are exactly four total, so all four lie in that window and are
> collinear on one axis.”

**Independent recomputation: choices, cadence, and legality.** After the first
post-S15 paired service, the real and shadow occupancies have sizes five and
seven. Thus `y_1` excludes at most `5+7` cells from a 217-cell
radius-eight ball, leaving at least 205. It is supported on both boards;
real S then has three stones, so every window through it has deficit at least
three and it is first-safe.

The first filler excludes the seven occupied shadow cells and
`T(y_1)`, leaving at least 209 candidates in the same-sized supported
ball. It is `Shat`'s fourth stone. The fixed old-debt certificate
`T(y_1)` is its fifth, so neither can terminate.

Before `y_2`, a conservative exclusion count is six current real cells,
nine shadow cells after including the already fixed certificate, and
`T^{-1}(z_1),T^{-1}(z_2)`: at most seventeen cells in a 217-cell ball.
The chosen `y_2` is fresh and supported and gives real S only four stones.
Because `sigma` is fixed and pure and the intervening shadow certificate
is already fixed independently of `y_2`, `z_1` and `z_2` are
well-defined strategy values. The carrier still queries them only at their
actual phases. Avoiding their inverse images keeps `T(y_2)` fresh, and
support from `T(y_1)` keeps it legal. Hence the look-ahead is the ordinary
counterstrategy use of a fixed pure strategy, not S12 preannouncement.

**Independent recomputation: `tau_E`.** An urgent window contains all
four real-S stones. Every urgent window must therefore lie on the *same
unique axis line*: two distinct axis lines cannot both contain the same four
distinct cells. Parameterize those cells by their indices on that line:

| Span | Length-six intervals containing all four | Hole-set transversal |
|---:|---:|---:|
| 5 | one | one cell |
| 4 | at most two | the shared internal missing cell, or at worst two choices |
| 3 | three, for four consecutive cells | `{-1,4}` hits `{-2,-1}`, `{-1,4}`, `{4,5}` |

F-blocked intervals only delete obligations. Thus `tau_E<=2` and the
canonical service pair exists. Its first shadow event is the fifth
`Fhat` stone and cannot win; its second yields counts six/five and exactly
the claimed S49/nonterminal dichotomy.

The artifact's conclusion is right. The unique-axis inference is merely
implicit before the one-dimensional span census.

**Proposed repair:** insert: “Since every urgent window contains the same
four distinct stones, all urgent windows use their unique common axis line.”

### 5. NOTE — S50.1's five-stone threat census is exhaustive

> “Its family of immediate winning cells has size at most two.”

**Independent recomputation.** With exactly five `Shat` stones, a
one-hole winning window must contain all five. If they are noncollinear, or
their collinear span exceeds five, there is no threat. For collinear indices:

- span five gives one length-six interval and one missing cell;
- span four means five consecutive stones and gives exactly the two shifted
  intervals, with outside holes at indices `-1` and `5`; and
- a smaller span is impossible for five distinct integer indices.

Thus there are at most two winning cells. In the two-threat case their
distance is six. Both are legal through the five-stone line. `Fhat`
starts the pair with four stones; after using both blockers it has six. Any
new six-window would have to contain both new cells, but a length-six engine
window has diameter five, so the blocking pair cannot itself win.

In the one-threat case, the blocker is only `Fhat`'s fifth stone.
Five `Fhat` stones in turn have at most two immediate winning cells. A
radius-eight support ball has 217 cells; the ten physical shadow occupancies
after the blocker plus at most two forbidden winning cells exclude at most
twelve. A different empty supported nonwinning padding coordinate exists.

**Proposed repair:** none.

### 6. MINOR — S51 is a finite-horizon cover, not a disjoint trichotomy

> “For every causal outer continuation attempt, within that shadow horizon at
> least one of the following occurs.”
>
> “raw preterminal `A_FS2^EV` membership or another mandatory branch
> obligation *other than the tested ... P5 ...* fails.”

**Independent recomputation.** S24 applies to the stated `h`. A legal,
reachable, nonterminal, `sigma`-consistent prefix of a strategy winning
from the root inherits winningness against every legal continuation: any
counterplay from `h` concatenates with the reached prefix. The compatible
nonterminal tree is finitely branching, so its depth has a finite bound in
single shadow placements.

If item 1 is avoided by an outer rule that continues supplying legal
append-only shadow play through that bound, a `Shat` win is impossible
under the alleged-winning premise. If item 2 is also avoided and
`Fhat` stays nonterminal through the bound, S24 is contradicted.
Therefore `Fhat` terminates; without a real sound stop, that is item 3.

The exclusion of the tested P5/(46.4) duty is honest. Definition 55.1's raw
class expressly retains the misaligned terminal outcome. An aligned event is
item 2; a shadow-only `Fhat` event is item 3. Neither ET nor LOCK is
silently assumed.

The three headings need not be mutually exclusive: an unrelated obligation
can fail at the same event as a real or shadow terminal result. A voluntarily
truncated live construction already falls under item 1 because it fails the
mandatory coverage/continuation obligation; no additional maximality premise
is needed. The proof therefore establishes an exhaustive non-disjoint cover,
not a partition in the strict sense suggested by “trichotomy.”

**Proposed repair:** rename S51 “finite-horizon stop cover,” or impose an
explicit first-applicable precedence to obtain a genuine trichotomy.

### 7. NOTE — S46's debt law is exact, and its non-preserving cases are not hidden

> “`a'(V,W)=a(V,W)-1_{k in W}+1_{z in V}`.”

**Independent recomputation.** Put `h=delta_H^F(V)`,
`r=delta_R^F(W)`, and `a=r-h`. The four incidences are:

| Paired-event incidence | `h'` | `r'` | `a'` |
|---|---:|---:|---:|
| `z notin V`, `k notin W` | `h` | `r` | `a` |
| `z notin V`, `k in W` | `h` | `r-1` | `a-1` |
| `z in V`, `k notin W` | `h-1` | `r` | `a+1` |
| `z in V`, `k in W` | `h-1` | `r-1` | `a` |

The dangerous cases are real. A shadow-only hit sends debt one to debt two.
A joint hit can take `h:2->1` while preserving `a=1`, which passes
the one-debt inequality but fails terminal readiness. Thus S46 is not an
automatic preservation theorem.

For every *retained* pre-event pair the equation is nevertheless exhaustive.
Neither F-owned append adds an opponent stone, so a pre-existing
opponent-unblocked window cannot become blocked during that F event. A
pre-event shadow deficit-three window containing `z` can newly enter the
near family at deficit two; a real deficit-four window containing `k` can
newly become an eligible assignment. An intervening S/`Shat` turn can
block old windows. Definition 55.2 handles all three facts honestly: it fixes
the current portfolio before querying `z_i`, recomputes after a
nonterminal first event, and completely reselects after the S-role turn.
Failure to find the next portfolio ejects the trace from the strict class.

**Proposed repair:** add this four-case table, especially the shadow-only and
joint-hit readiness failures, as an exposition aid. No theorem change is
needed.

### 8. NOTE — `F-CAD_2^st` is a genuine many-window-to-one cover, not a point map

> “The assignment may be many-to-one.”
>
> “This is the maintained state invariant. It does not itself say which real
> service coordinate is selected.”

**Independent recomputation.** The domain is finite: every shadow window
with deficit at most two contains at least four of the finitely many
`Fhat` stones, and each stone belongs to eighteen engine windows. For an
admissible assignment,

`delta_R^F(W) <= delta_H^F(V)+1 <= 3`,

so `W` contains at least one real-F stone and lies in the finite union of
the eighteen-window stars of the finite real-F set. Hence the finite
portfolio order and least admissible selection are legitimate.

The policy maps windows to windows, not stones to coordinates; it has no
injectivity condition; and several `V` may use the same `W`. S41
actually uses that freedom when `V={(1,0),...,(6,0)}` and the shifted
`V_2={(2,0),...,(7,0)}` share one real `W`. “One-debt” bounds each
assigned scalar debt; it does not assert that only one portfolio entry may
have debt one. The inherited S-role certificate still uses `T`, but no
F-role point map is imported into CAD or event pairing.

**Proposed repair:** none; detached summaries should keep “no point map”
scoped to the F-window/event module.

### 9. MINOR — S41's pathwise assignments should be explicitly totalized into a fixed policy

> “These prefix-indexed assignments define one causal `Pi` on the
> displayed finite trace.”

**Independent recomputation.** The displayed assignments are pre-query and
depend only on their reached prefixes, so there is no causal defect on S41.
Definition 55.2, however, first says to fix a portfolio policy before the
coupled trace. If “policy” means a total pure rule on all physical prefixes,
the proof gives only its finite on-path table. The missing extension is
immediate but should be stated: special-case the finitely many S41 prefixes
and, on every other prefix, return the least admissible portfolio under the
already fixed orders.

That rule is fixed independently of which branch is later followed and
reproduces every displayed S41 choice. To keep the type honest at prefixes
with no admissible portfolio, formalize the policy as
`Option<Portfolio>` (or an equivalent partial admissibility selector):
`None` is then the strict-class exit. The issue is formal presentation,
not a failed existence, census, or causality argument.

**Proposed repair:** append the off-path totalization just described to S48
and make failure explicit in the policy's codomain rather than treating
“exit” as a portfolio value.

### 10. NOTE — CAD plus LOCK yields an actual co-terminal append; the augmented rule stays conditional

> “`A_FS2^{CAD+LOCK}(sigma) subseteq A_FS2^ET(sigma)`.”
>
> “If a required nonterminal admissible set is empty, the augmented branch
> exits its strict class.”

**Independent recomputation.** Immediately before a shadow-terminal
prescription, each completed witness `V_*` has shadow deficit one and the
prescription is its unique empty cell. Readiness gives

`delta_R^F(pi_i(V_*)) <= delta_H^F(V_*) = 1`.

The real board is live, so the real deficit is not zero; it is exactly one.
Because the assigned real window is S-unblocked, its sixth cell is physically
empty. LOCK puts the actual legal canonical `k_i` in that window, hence
at its unique hole. The real append completes the physical six on the same
atomic event. CAD alone proves only readiness; CAD plus LOCK proves inherited
event-terminal alignment. The implication direction is exact.

For the augmented rule, the terminal hole cannot be occupied or illegal:
five real-F stones occupy the rest of its length-six line, placing the hole
within distance at most five of support. At a second placement it also cannot
equal the already occupied first service cell. The nonterminal clauses are
not availability assertions: the first requires a legal cell that leaves a
residual one-cell urgent transversal and a post-event CAD portfolio; the
second requires the unresolved service or legal padding. An empty candidate
set is an explicit strict-class exit. The second prescription is queried only
after both first appends are known nonterminal.

Therefore S47 part 1 gives actual alignment, and part 2 gives it for every
trace admitted through its terminal event. No complete terminal execution of
the exact augmented least-choice handler is supplied; its `OPEN` label is
correct.

**Proposed repair:** none.

### 11. NOTE — S51.1 draws the branch-(C) circularity boundary correctly

> “Branch (C) is therefore a valid successful contradiction stop, but it
> cannot be used as a continuing device while still calling `sigma`
> alleged-winning.”

**Independent recomputation.** At a genuine reachable common-live
`Shat` node, suppose an `Fhat`-unblocked window has deficit at most
the one or two `Shat` placements remaining in the turn. Its holes are
physically empty, are within distance at most five of the existing stones in
the same line, and remain legal sequentially. Filling them produces an actual
physical shadow-board `Shat` six. Those finite on-path choices extend by
least-legal off-path moves to a total counterstrategy, contradicting that `sigma`
wins against every counterplay. This is exactly inherited S39, including its
common-live and alleged-winning hypotheses.

It is therefore invalid to obtain a convenient `Shat` win, continue the
shadow history past terminality, and still invoke winningness. It is not
circular to feed `sigma` legal nonwinning `Shat` moves and ask it to
meet its finite horizon. Only an actual `Shat` win, or an infinite
nonterminal compatible play, is the refutation.

**Proposed repair:** in section 61, “directly refutational” is clearer than
“circular/directly refutational”; circularity arises only if the terminal
branch is reused as a continuing construction.

### 12. NOTE — S52's rolling exit partition is complete at its checkpoint-local scope

> “At each indicated checkpoint the following cases partition the possible
> continuation.”

**Independent recomputation.** For a legal post-placement `S@y`,
`d_y=0` exactly when an F-unblocked window through `y` contains six
real-S stones, so the just-made append is terminal. At `FirstStone` one
placement remains; the strict shield is integer
`delta>1`, giving the disjoint cases terminal `0`, unsafe nonterminal
`1`, and safe `>=2`. At `SecondStone` no S placement remains;
`0` is terminal and every `>=1` is nonwinning. The defined value
`infinity` belongs to the `>=2` case.

For old debt `u=T(e)`, physical owner-disjoint occupancy gives exactly
`Shat`-occupied, `Fhat`-occupied, or empty. An empty coordinate is
legal or illegal; a legal append is winning or nonwinning. Those are exactly
the five listed cases. Correct-role occupancy, wrong-role occupancy, and
fresh-illegality are distinct physical states, not labels. A fresh legal win
is a terminal module result, not a live membership failure.

After a nonwinning S pair, Definition 38.2 admits exactly
`tau_E<=2`; the complement exits before canonical service absent an outer
repair. For a paired F event, the two binary terminal predicates give the
four rows listed in S52. Round-6 section 53 makes the double-terminal row
coherent even when either first placement would end its individual board.

The numbered partitions are successive checkpoint partitions, not a claim
that all later handler events are one disjoint leaf set. In particular, an
old-certificate or filler append that wins remains the inherited physical
module stop. It is not a missing “unavailable” case. If debt is empty, a
legal filler always exists on `Z^2`: from an occupied cell of maximal
q-coordinate, `a+(1,0)` and `a+(1,-1)` are distinct empty legal
neighbors, and at most one is the forbidden `T(y)`.

**Proposed repair:** state parenthetically that `infinity>=2`. No
partition repair is required.

### 13. NOTE — S53 and S53.1 reconcile exactly the correct-role occupied class

> “Physically recognize the already-present stone: move `e` from
> `E_S` to `C_S`.”
>
> “The urgent family becomes empty and the recomputed transversal number is
> zero.”

**Independent recomputation.** Under the premise
`Shat@T(e)` already exists, that stone is precisely the physical
certificate required by Definition 30.1. Reclassifying `e` before the
next real query changes no board, owner, legal store, actor, or phase and
removes every old E-live window. It also avoids a transient two-debt label.
If done after `S@y` instead, the inherited calculation

`delta'(W)-m'=(delta(W)-m)+1-1_{y in W}>0`

shows that no shield violation was concealed.

After the observed guarded nonterminal `y`, the branch dispatch is
exhaustive:

1. if `T(y)` is fresh and legal, appending it is exact physical branch A;
   a nonwin certifies `y` immediately. A win is a physical module stop
   generally, and is branch (C) and the direct contradiction only under a
   genuine alleged-winning, `sigma`-consistent outer premise; or
2. otherwise a fresh legal filler distinct from `T(y)` exists by the
   maximal-q argument. A nonwinning filler leaves only `E'_S={y}`, and
   the first-safe/nonwinning guard gives the precise one-step deadline
   shield. A winning filler is a physical module stop and becomes a
   counterstrategy contradiction only under the genuine alleged-winning,
   `sigma`-consistent outer premise.

One physical S-role stone is appended on each board, so two nonwins reach
matching `SecondStone` phases after a first coordinate and matching F
`FirstStone` phases after a second. The theorem does not claim future
certificate availability in branch B. Wrong-role occupancy cannot be
reconciled because the physical owner is wrong; a fresh unsupported image
still lacks legal support.

At S53.1's F checkpoint, reclassifying the singleton makes
`E_S=empty`. Since every E-live and urgent window must meet `E_S`,
the urgent family is empty and the empty set is a transversal of exact size
zero. This is physical reconciliation, not the certificate discount excluded
by S44.

**Proposed repair:** none.

### 14. NOTE — every coordinate and cadence claim in S54's cylinder is legal

> “S54 proves only that changing `T` inside the pair can move a persistent
> cut endpoint ... and can close the exact two episodes on one physical
> cylinder.”
>
> “Full `C_A^{K=2/pair}` remains **OPEN**.”

**Independent recomputation.** The initial S15 representation under
`T_1(q,r)=(q+1,r-1)` maps the real opener to
`Fhat@(1,-1)` and the two real-S cells to
`Shat@(1,0),(2,0)`; `Shat@(0,0)` is the shadow opening and
`Fhat@(0,-1)` the proxy. The later real pair
`F@(1,0),(2,0)` maps to the genuine reached pair
`Fhat@(2,-1),(3,-1)`. Both boards are then at S
`FirstStone`.

The rebinding census is:

| Binding stage | Represented sets | Physical complement | Next physical pair |
|---|---|---|---|
| `T_0(q,r)=(q,r-1)` | real F -> `(0,-1),(1,-1),(2,-1)`; real S -> `(0,0),(1,0)` | `Shat@(2,0)`, `Fhat@(3,-1)` | real `c_1=(2,1)`; shadow filler `(3,0)` |
| restored `T_1` | real F -> `(1,-1),(2,-1),(3,-1)`; three real S -> `(1,0),(2,0),(3,0)` | `Fhat@(0,-1)`, `Shat@(0,0)` | real `c_2=(-1,0)`; shadow filler `(-1,-1)` |
| restored `T_0` | all three real-F and all four real-S images present | `Fhat@(3,-1)`, `Shat@(3,0)` | common F `FirstStone` |

`c_1` is adjacent to `S@(1,1)`, `(3,0)` is adjacent to
`Shat@(2,0)`, `c_2` is adjacent to the real opener, and
`(-1,-1)` is supported within distance two of the shadow opening.
Every coordinate is fresh. No owner reaches six stones, so all four episode
appends are nonterminal and phase matching is exact.

The negative control is equally important. Under fixed `T_0` the actual
S43.1 first cut would be `(3,0)`, because
`Fhat@(3,-1)` is adjacent to represented `Fhat@(2,-1)`. The displayed
`c_1` instead maps to already present `Shat@(2,0)`, its filler becomes
the proxy, and `c_2` maps exactly to `Shat@(-1,-1)`. Fixed
`T_0` reaches the same final complement. Hence the cylinder proves finite
consistency only: no necessity of rebinding, cut escape, recurrence,
arbitrary-S response, universal P3/P5, or globally winning strategy.

**Proposed repair:** none.

### 15. MINOR — section 59 again drops two required carry-forward caveats

> “Round-6 review's authoritative twelve obstacles.”
>
> “Round-4 review's ten-item agenda.”

**Independent recomputation.** The round-6 hostile review's Finding 12
restored four exact caveats. Round 7 carries only two at the required local
ledger sites:

| Restored round-6 caveat | Round-7 local disposition |
|---|---|
| total nonisometric zero-lag point recodings survive | retained in §59.2 row 1 |
| common-only real wins and simultaneous legality/terminal maintenance remain duties | common-only real wins disappear from §59.2 row 10; “simultaneous P2/P3/P5R” is not the same explicit legality plus F-terminal-maintenance statement |
| S13 belongs in agenda row 2 | omitted again from §59.3 row 2; it appears only later in row 6 and the regression matrix |
| S14 belongs in agenda row 3 | retained in §59.3 row 3 |

The omissions do not silently prove either obligation. Section 54.3 itself
states the common-only-win and simultaneous-maintenance duty, section 54.4
binds S13, and section 60's regression matrix still calls S13 open at its
premises. The mathematical status is conservative, but the tables advertised
as authoritative carry-forwards are not literally complete.

**Proposed repair:** in §59.2 row 10, restore “common-only real wins” and
“simultaneous legality and terminal maintenance,” including P5 rather than
only the P5R abbreviation. In §59.3 row 2, restore the explicit S13
fixed-isometry FIFO-frontier sentence.

### 16. MINOR — section 62 omits the landed artifact identity

> “Requested input state ... `09e27a93`.”
>
> “During the session a read-only check observed the branch reference at
> `a8a0b92d`.”

**Independent recomputation.** Both statements are consistent with Git.
`09e27a93` is the named authoring input; `a8a0b92d` is the observed
intermediate descendant and immediate parent of the landing. The artifact,
however, does not record that it actually landed at
`798bfb75cee282960640fd8b4abcbfea47c11404`. It is first added there,
along with the authoring prompt, and the current copy is byte-identical to
that landing's blob
`2241724248da0bb66b01d7d1764496b1562f0fb5` (SHA-256
`12e228fe0d8aba68fe58ff090960d522627116ac194644ce3af58df4e8796140`).

The intermediate name-only difference did include unrelated `GAP_RAW`
artifacts, as round 7 says. They were not used by this review, and no theorem
in round 7 cites their contents. The defect is the absent landing record, not
a provenance contradiction or mathematical dependence.

**Proposed repair:** add the full landing commit, blob, and SHA-256 to §62.1
and distinguish the named input, observed intermediate parent, and
first landed artifact.

### 17. NOTE — inherited regressions and outcome boundaries remain substantively binding

> “**Global target:** `NL_F` remains **OPEN**.”
>
> “A strict-subclass carrier and a conditional negative barrier do not select
> a determinacy alternative.”

**Independent recomputation.** Round 7 respects the operative content of all
named regressions:

- S13 still excludes every fixed-isometry one-stone FIFO scheme at its
  premises; S54 neither uses that schedule nor proves separation from fixed
  `T_0`.
- S14 and S25 remain the terminal-memory tests outside the guarded/reconciled
  lag classes.
- S18 remains binding for spatial inverse/FIFO proposals; the F-role event
  carrier makes no reverse-legality inference.
- S20 remains the universal proxy-assisted `Fhat` terminal-fidelity duty;
  S47 solves it only under CAD+LOCK or admitted augmentation, while S49 proves
  an exact impossible count regime.
- S30's `tau_E=5` and S31's six-blocker installation cost remain open
  generally; S53.1 covers only an already-certified singleton.

Section 53's atomic final-event semantics is used consistently. Every new
finite history is append-only and strategy-own at its claimed scope. No
result asserts universal `A_FS2` admission, global event alignment, full
per-pair `K=2` success, or either determinacy alternative. D2 remains only
the bridge to `NL_F`.

**Proposed repair:** only Finding 15's local ledger restoration.

## Per-theorem verdicts

| Result | Source status | Review verdict | Exact disposition |
|---|---|---|---|
| Production rule contract | PROVEN inheritance | **CONFIRMED** | Physical emptiness, radius-eight support, sequential insertion, per-append terminality, terminal no-continuation, append-only history, and three six-window axes match production |
| Definition 55.1 `A_FS2^EV` | Definition | **CONFIRMED** | Removes (46.4), retains the section-53 atomic final event, and classifies all four F-role terminal combinations without converting membership exits into wins |
| S46 debt update | PROVEN | **CONFIRMED** | Exact for every retained opponent-unblocked pair; does not preserve CAD automatically, and the source makes reassignment/admission an explicit premise |
| Definition 55.2 `F-CAD_2^st` | Definition | **CONFIRMED** | Pre-query causal, finite, many-window-to-one, no F-role point map; one-debt and readiness are separate inequalities |
| Definition 55.3 `F-LOCK` | Definition / obligation | **CONFIRMED** | Residual canonical terminal cell must lie in an assigned completed window; sufficient but not necessary and not derived from S46 |
| S47(1) canonical CAD+LOCK transfer | PROVEN | **CONFIRMED** | Readiness makes the assigned real window one-hole and LOCK makes canonical service fill that hole on the same event |
| S47(2) augmented exact alignment | PROVEN conditionally | **CONFIRMED AT RECURRING-ADMISSION SCOPE** | Terminal hole is empty/legal; nonterminal service and next portfolios remain explicit admissibility tests |
| Complete augmented least-choice terminal trace | OPEN | **CONFIRMED OPEN** | S41 is canonical CAD+LOCK, not a demonstrated complete execution of the distinct augmented selector |
| S48 / S41 debt audit | PROVEN | **CONFIRMED-WITH-MINOR-FORMALIZATION** | Census and repayment are exact; explicitly totalize the finite on-path `Pi` table (Finding 9) |
| S48 / S42 readiness audit | PROVEN | **CONFIRMED** | Its legal live `A_FS2` handler prefix reaches shadow deficit one while every real window has deficit at least two before the raw terminal extension |
| Universal canonical CAD maintenance | PROVEN negation | **CONFIRMED** | S42 refutes preservation through every live canonical handler extension from `A_FS2`; the negation is not inflated to every prefix or every selector |
| S49 sixth-`Fhat`-stone barrier | PROVEN | **CONFIRMED** | Fourth post-S15 one-for-one event leaves counts six/five; every two-cell selector or augmentation within that scope fails co-terminal count at a sixth-stone terminal event |
| S49.1 universal canonical-invariant negation | PROVEN | **CONFIRMED** | S42 defeats a property preserved on every legal pure-strategy canonical extension; alleged-winning-only and CAD-restricted properties survive |
| S50 adaptive earliest-cycle dichotomy | PROVEN | **CONFIRMED-WITH-MINOR-EXPOSITION** | All choices, counts, phases, legality, certificate freshness, and `tau_E<=2` recompute; common-axis inference should be stated (Finding 4) |
| S50.1 five-`Shat`-stone control | PROVEN | **CONFIRMED** | At most two immediate holes; the distance-six double block cannot make an `Fhat` six; one-threat padding exists |
| S51 finite-horizon stop result | PROVEN | **CONFIRMED-WITH-MINOR-TERMINOLOGY** | S24 scope and P5 exclusion are exact; the alternatives form an overlapping cover, with voluntary truncation classified under item 1 (Finding 6) |
| S51.1 branch-(C) boundary | PROVEN | **CONFIRMED** | A reachable physical `Shat` win is already the counterstrategy contradiction and cannot be continued |
| S52 rolling exit partition | PROVEN | **CONFIRMED** | All five indicated checkpoint partitions, filler existence, and the atomic F-event table are exact |
| S53 occupied-certificate reconciliation | PROVEN | **CONFIRMED** | Correct-role physical occupancy clears old debt; branch A/B dispatch, guards, filler, terminal cases, and phase restoration are legal |
| S53.1 high-transversal reconciliation | PROVEN | **CONFIRMED** | An already-present correct-role singleton certificate makes `E_S=U_E=empty` and `tau_E=0` |
| S54 alternating-translation cylinder | PROVEN finite execution | **CONFIRMED** | Every coordinate, support, owner image, complement, cadence, and nonterminal count is exact |
| Full per-pair `K=2` success class | OPEN | **CONFIRMED OPEN** | Fixed `T_0` closes the same cylinder; S54 gives no cut separation, arbitrary response, recurrence, P3, or terminal theorem |
| Section 59 authoritative ledgers | ledger maintenance | **CONFIRMED-WITH-MINOR-ERRATA** | Mathematical statuses remain open, but common-only/simultaneous maintenance and S13-at-row-2 are locally omitted (Finding 15) |
| Section 62 provenance | documentation | **CONFIRMED-WITH-MINOR-ERRATA** | Named input and intermediate are accurate; landing `798bfb75` and artifact hashes are absent (Finding 16) |
| Universal alignment for alleged-winning `sigma` | OPEN | **CONFIRMED OPEN** | Neither universal CAD+LOCK nor a forced strategy-own fast/later misalignment is proved |
| Global `P0–P6/P5R` coupling | OPEN | **CONFIRMED OPEN** | The local branch system still does not cover every legal real-S continuation and every genuine prescribed F event |
| `NL_F` | OPEN | **CONFIRMED OPEN** | D2 remains only the determinacy bridge; neither global alternative is selected |

No result receives **REFUTED** or **MAJOR**. The five **MINOR** findings are
formalization, exposition, terminology, ledger, and provenance repairs;
none changes a coordinate, deficit, count barrier, partition, or finite
execution.

## Exact unresolved obstacles after review

The authoritative open state, with the two section-59 omissions restored, is:

1. **Full per-pair and broader zero-lag branch (A).** S54 is one finite
   `T_0/T_1/T_0` execution and has a fixed-`T_0` realization. It gives
   no arbitrary-S response or recurrence. Intra-pair changing isometries,
   total nonisometric zero-lag point recodings, non-total/window recodings,
   and indefinitely one repair per placement remain open.
2. **Pre-checkpoint and recurring P3 coverage.** S50 reaches the next two
   actual prescriptions for every fixed strategy after S15. No theorem
   reaches every later first and second prescription while preserving one
   genuine legal history, common phase, serviceability, and terminal rules.
3. **Coverage outside strict `A_FS2`.** S53 covers correct-role occupied
   old certificates. Nonterminal first-unsafe cells, real-S terminal cells
   without a same-step certificate, wrong-role occupancy, fresh unsupported
   certificates, uncertified `tau_E>2`, and shadow-terminal events outside
   CAD+LOCK/augmented admission remain open.
4. **P5R through every lag and recode.** Every real-only S stone must stay
   shielded, become physically certified, be blocked by real F, or receive an
   actual same-step `Shat` terminal reflection. S14 and S25 remain
   binding, and common-only real wins still require an outer physical
   transfer; they cannot be dismissed by labels.
5. **Canonical and augmented F-service compatibility.** CAD is only a state
   admission and LOCK is a separate canonical selector duty. S41 witnesses
   one canonical terminal trace; S42 shows universal maintenance is false.
   No complete terminal trace for the exact augmented least-choice rule is
   proved, and no rule supplies its nonterminal candidates and portfolios
   through the S24 horizon.
6. **Universal shadow-`Fhat` terminal fidelity.** S47 is conditional.
   S49 forbids alignment at terminal age six, but no alleged-winning
   `sigma` is forced to terminate at that age. Every later first- or
   second-placement S20-type terminal prescription still needs a same-event
   real certificate or a strategy-own misalignment proof.
7. **Reverse legality for spatial carriers.** Temporal event pairing avoids
   inversion. Every inverse-map or fixed-FIFO proposal still owes S18, S13,
   and the sequentially updated unsupported and collision sets.
8. **Strategy domain and physical persistence.** Every filler, proxy,
   service cell, certificate recognition, queue rotation, and rebinding must
   remain one genuine legal append-only shadow history whose `Fhat` moves
   agree with the total strategy `sigma`. Old stones retain occupancy,
   support, blocking, and terminal-window effects.
9. **Global causality.** The new event selectors and S50 counterstrategy are
   causal at their scopes. Every outer repair, future backing coordinate, or
   spatial recode must still avoid fixing an exposed real-F cell across an
   intervening S turn as in S12.
10. **Universal window-certificate maintenance.** S46 updates retained pairs
    only. New windows, reassignment after arbitrary S turns, canonical LOCK,
    common-only real wins, and simultaneous legality plus P2/P3/P5/P5R
    terminal maintenance need one recurring physical handler. The shorthand
    “simultaneous P2/P3/P5R” does not discharge the omitted P5/common-win
    duties.
11. **High-transversal service and permanent fencing.** S53.1 handles only an
    already-certified singleton. S30 still has exact `tau_E=5` and S31
    still costs six permanent blockers; availability, installation under
    interruption, S occupation, reconciliation, and P3 compatibility are
    open.
12. **Strategy-specific reachability and outcome.** S50 is adaptive for every
    fixed strategy but only reaches the fourth post-S15 prescription. S51
    reduces the later route to a controller that preserves membership and
    avoids a real sound stop through a strategy-dependent finite horizon; it
    does not build that controller. Until every alleged-winning strategy is
    refuted or one global carrier is completed, universal alignment, the
    global coupling, and `NL_F` remain open.

## Overall verdict and section dispositions

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** Hostile recomputation found no
false debt equation, hidden injectivity, illegal terminal hole, wrong S41/S42
census, vacuous count barrier, bad cadence, missed five-stone threat, invalid
S24 use, incomplete rolling checkpoint partition, illegal reconciliation, or
bad S54 coordinate. Five narrow repairs remain, led by section 59's repeated
carry-forward omissions and the fixed-policy formalization.

1. **Section 55 — CONFIRMED CONDITIONAL POSITIVE, WITH ONE PROVEN NEGATION.**
   S46 is the exact per-pair update but not a preservation theorem.
   `F-CAD_2^st` is honest many-to-one state readiness; LOCK is the
   separate canonical coordinate duty. S47 gives real co-terminal alignment
   on CAD+LOCK and recurring augmented admission. S48 confirms S41 and proves
   with S42 that canonical service does not preserve CAD through every live
   canonical handler extension from `A_FS2`. The complete augmented
   terminal trace remains open.
2. **Section 56 — CONFIRMED PARTIAL NEGATIVE ROUTE.** S49 is a substantive,
   selector-independent six-versus-five barrier at its one-for-one scope.
   S50 reaches that test for every pure strategy but cannot force the fourth
   prescription to win; S50.1 blocks the naive earliest threat coercion.
   S51 is a valid finite-horizon non-disjoint cover; its “trichotomy” title
   needs only the minor terminology repair, and S51.1 states the direct
   branch-(C) contradiction exactly.
3. **Section 57 — CONFIRMED ONE NEW EJECTION CLASS.** S52's checkpoint
   partitions are exhaustive. S53 physically reconciles a correct-role
   occupied old certificate and dispatches legally to exact branch A or a
   guarded one-step branch B; S53.1 makes an already-certified singleton's
   urgent family empty. The expressly listed remainder stays open.
4. **Section 58 — CONFIRMED FINITE CONSTRUCTION ONLY.** S54 is a genuine
   legal two-episode alternating-translation cylinder. Fixed `T_0` also
   closes it, so it proves neither rebinding necessity nor escape from
   S43.1's fixed-map subclass. Full per-pair `K=2` remains open.

**Most severe finding:** Finding 15, **MINOR**. The supposedly authoritative
carry-forward again omits common-only real wins and simultaneous
legality/terminal maintenance from obstacle 10, and again drops S13 from
agenda row 2; the obligations remain open elsewhere but must be restored
locally.
