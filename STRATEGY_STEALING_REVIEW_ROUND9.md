# R-ST9-REV — Hostile review of `STRATEGY_STEALING_ROUND9.md`

## Method and proof boundary

**Reviewed artifact.** `STRATEGY_STEALING_ROUND9.md` first appears at landing
commit `e19e97f73f66154459500be1d578bb0a5a807592` on branch
`hunt/gap-raw`. Its immediate parent is the task-declared input
`9e57ea060462788841d1f8f761db894981b482e9`. The landed Git blob is
`9c999896f1e3f64aeb60bcbc3e4ffb16e181081c`; the worktree copy has that same
blob identity and SHA-256
`d2b718a173601a8704c10a8b9929e1f94bb72b53198a3900402f1dc302b4c31d`.
During final assembly the shared branch advanced externally to
`f5349d3eb985cdb9ee719ec75272f3a73772604d`. The requested landing remains
its ancestor, the Round-9 blob stayed identical, and none of the reviewed
Round5-Round9 or engine-source paths changed; this audit remained pinned to
`e19e97f7`.

Default posture was **REFUTE**. I read the required evidence in the prescribed
order and in full:

1. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, then
   `STRATEGY_STEALING_REVIEW_ROUND5.md`;
2. `STRATEGY_STEALING_ROUND6.md`, including binding section 53 and the
   terminal-closure semantics, then `STRATEGY_STEALING_REVIEW_ROUND6.md`;
3. `STRATEGY_STEALING_ROUND7.md`, including binding section 63, then
   `STRATEGY_STEALING_REVIEW_ROUND7.md`;
4. `STRATEGY_STEALING_ROUND8.md`, including rewritten section 68 and binding
   section 73, then `STRATEGY_STEALING_REVIEW_ROUND8.md`; and
5. `STRATEGY_STEALING_ROUND9.md`.

Only after that corpus read, I read the six cited production sources in full:
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
The audit uses their physical radius-eight legal store, sequential
`Opening -> FirstStone -> SecondStone` transitions, append-before-win test,
eighteen incident Q/R/QR windows, immediate terminality, and append-only
forward histories. I did not open or use any `GAP_RAW_*` file as evidence.
The name-only difference from `ab0fd965` to `9e57ea06` was used solely for the
provenance audit and shows no strategy-stealing or engine-source change.

No Cargo command, Lean command, proof harness, test harness, executable game
search, solver, or proof-search program was run. Every count, exclusion bound,
window census, cadence transition, deficit identity, and obstruction below was
recomputed by hand. No commit was created, and unrelated workspace entries
were left untouched.

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** S70's three-way disposition is
proved at its deliberately constructed selected-history, next-entrance scope;
S70.1 is a genuine optional selected-history real-F closure; S70.2 forces the
line-seeded cliff for `FAST_8^{S15}` while retaining the carrier-versus-`sigma`
distinction; and S69 is exact for scalar `RES_1`. No named reserve, ordinary
two-append, section-53 closure, or terminal-moment S63 conversion repairs the
fast line. The sole required defect found is provenance: section 83 does not
record the task-specified input/landing pair and in fact describes a different
proof-input chronology. That defect does not change a mathematical verdict.

## Numbered findings

### Finding 1 — NOTE: the production cadence and terminal grammar used by Round 9 are exact

> “After the opening, a turn has a sequential `FirstStone` and
> `SecondStone`.”

**Independent recomputation.** The engine begins with `Player0` in `Opening`.
A legal nonwinning opening passes to `Player1 FirstStone`; a nonwinning first
normal append retains the owner at `SecondStone`; and a nonwinning second
append passes to the opponent at `FirstStone`. Thus the physical cadence is

`F ; S,S ; F,F ; S,S ; F,F ; ...`.

The board inserts the stone, updates the legal store and all eighteen incident
windows, and only then tests for a six. A win sets terminality before any phase
advance, and terminal states expose no legal action. Section 53's final paired
event can therefore contain the two already-associated cross-board appends,
but it grants no third placement and cannot continue either engine afterward.
These are exactly the rule facts used in S64-S70.

**Proposed repair:** none.

### Finding 2 — NOTE: S64's reserve census is exact and excludes the named reserve from the fast cylinder

> “The fast line fails `RES_1` at every `q_j` and is **never admitted** to
> `R_1` at either eligible `FirstStone` entrance `q_1` or `q_3`.”

**Independent recomputation.** S15 starts the F roles at shadow/real counts
`(2,1)`. Each genuine one-for-one F event raises both counts once, so the
four pre-query counts are `(2,1)`, `(3,2)`, `(4,3)`, and `(5,4)`. If `z_4`
is the sixth Fhat stone and completes a window `V`, all six Fhat stones lie in
`V`; no Shat stone blocks it. At `q_1,q_2,q_3,q_4`, respectively, `V`
therefore contains two, three, four, and five Fhat stones and has deficits
`4,3,2,1`. The total-stone inequality `mu_P>=6-|X_P|` supplies the same
lower bounds, making those shadow minima exact. It supplies real lower bounds
`5,4,3,2`.

The resulting scalar table is therefore:

| Query | `(mu_H, lower bound for mu_R)` | `RES_1` requirement | Verdict |
|---|---:|---|---|
| `q_1` | `(4,>=5)` | `mu_R<=2` | fails |
| `q_2` | `(3,>=4)` | `mu_R<=2` | fails |
| `q_3` | `(2,>=3)` | `mu_R<=2` | fails |
| `q_4` | `(1,>=2)` | `mu_R=1` | fails |

The future winning window also has exactly the holes `z_3,z_4` at `q_3`, so
`z_3` hits a deficit-two shadow window and causes
`(2,mu_R>=3)->(1,mu_R'>=2)`. On the `mu_R=3` slice this is S58 exactly; on
the larger slice S55 says the scalar CAD premise was absent already. There is
no earlier reserve state to “eject.” Importing S59's prepaid witness would
import extra persistent real-F stones or an unmatched append, so S64.1's
same-history splice obstruction follows. This does not deny the nonempty S59
class on different histories.

**Proposed repair:** none.

### Finding 3 — NOTE: two ordinary appends and section 53's closure both fail for the fast terminal

> “They change the real-F total from three to five; no ordering or spatial
> choice can make five stones fill a six-window.”

**Independent recomputation.** At `q_3` real F has exactly three stones. The
ordinary second post-S15 F turn supplies exactly `k_3` and `k_4`, ending with
five. A six-cell F window therefore cannot be full. Inserting `k_4` before
the `z_4` query does not change that count and abandons the stipulated common
phase and one-event-per-microstep carrier. After the nonwinning second real
append, real play passes to S; a third F append would be illegal. On the
shadow side, `z_4` terminates the engine, so there is no later shadow phase.

Binding section 53 changes only the bookkeeping order of the already-paired
real and shadow appends in a final coupled event. Under Definition 46.1 it
creates no physical stone, does not continue a terminal engine, and cannot
turn five real F stones into six. It is sound when the requisite physical
terminal facts already exist; it is not a source of those facts. Thus the
ordinary-two-append and named-closure failures are independent and both are
proved.

**Proposed repair:** none.

### Finding 4 — NOTE: S63 has the wrong role and premise at the fast terminal microstep

> “S63 is an S-role direct-refutation stop ... that premise is not forced
> merely by reaching the S49 F event.”

**Independent recomputation.** S63 requires a common-live S-role,
mirror-clean, first-unsafe event with two fresh legal image holes. The fast
segment queried at `q_3`, appended the first F/Fhat pair, queried at `q_4`,
and then reached the Fhat terminal on the second Fhat placement. It is an F
turn throughout. The preceding S pair in S50 is first-safe and nonterminal,
so it supplies no deferred S63 premise. At the terminal node common liveness
is gone and no continuation can manufacture one.

S70.1 is not a hidden conversion of this branch. It modifies the constructed
positive horn in advance by protecting `W_0`, reaches a different live F
query, and then installs the already-available sixth real-F stone. On the
fast branch S70.2 instead stops at the prior S58 carrier cliff. Hence none of
S63, S70.1, or their combination repairs the S49 event at the claimed scope.

**Proposed repair:** none.

### Finding 5 — NOTE: the remaining new lemmas and the pre-S70 partition are sound at their stated scopes

> “This is intentionally not a synthesis theorem.”

**Independent recomputation.** S65.1 starts after the misalignment with a
terminal shadow board and a live real board at S `FirstStone`, containing
five F and four S stones. Every immediate real-F winning window must contain
all five F stones. Five distinct cells lying in a six-window span either four
consecutive steps, giving at most the two exterior holes, or span five,
giving the sole internal hole. Thus there are at most two such holes. S can
occupy them during its legal pair (or stop earlier by winning); supported
filler cells exist when fewer than two holes occur: a radius-eight support
ball has 217 cells, while the current occupancy and the at-most-two new
S-winning completion cells exclude only finitely many of them. This is only
a negative control on existing one-move threats, not a later-game result.

S66's timing follows from the same count lower bound: `RES_1` fails at
`q_1,q_2,q_3`; at `q_4` it is possible only with `mu_H>=2,mu_R=2`, but
`q_4` is `SecondStone`. An S59-style entrance can therefore first occur only
after the second F pair and the following S pair. S67 correctly reads slow
depth only as existence of a genuine
`sigma`-consistent shadow branch live through local placement eight. It does
not infer a legal real lift. S67.1 is the direct S55-S58 scalar trichotomy at
`q_3`: `(2,>3)` has no CAD; `(2,3)` has CAD but no reserve and splits into
S58/S57/residue by the actual pair; `(>=3,>=3)` has vacuous near-window CAD
but no inherited prepayment theorem.

Finally, S68's five labels are exhaustive only after an extension attempt is
already fixed: terminal versus live; then first S58 event versus none; then a
mandatory interface failure versus arrival; then `R_1` membership versus its
live complement. Its priority makes the labels disjoint but supplies no
selector. Round 9 explicitly leaves the misaligned terminal, strict ejection,
and live-complement classes as residues until S70 constructs its separate
line-seeded selected history. The theorem therefore does not repeat the
refuted Round-8 synthesis.

**Proposed repair:** none.

### Finding 6 — NOTE: the binding errata and global quantifier boundaries are preserved

> “S70 supplies one causal candidate-own first-cycle history for every fixed
> `sigma`; it bypasses S67's branch-selection gap but proves neither
> completion nor indefinite recurrence of the next reserve cycle.”

**Independent recomputation.** The round uses section 44's transient-debt
inequality rather than treating bookkeeping as physical occupation; section
53 and Definition 46.1 only as terminal-event bookkeeping; section 63 only
on its exact S-role premise; and S49 as the genuine sixth-stone barrier. S59
is invoked only for the quiet/`R_1` corridor with CAD plus augmented
`F-LOCK^+`; canonical `F-LOCK` remains open. Every conclusion drawn from the
S58 cliff, including its use in S60/S70.2, is described as failure of the
one-for-one carrier, never as defeat of `sigma`.

The quantified S70 statement is `for every fixed pure sigma`, followed by a
rule constructed from that fixed strategy's prescriptions. It is not one
strategy-independent rule uniform over all `sigma`, and it does not quantify
over arbitrary real-S continuations. All actual choices use reached-prefix
data or the accepted pure-strategy prospective computation; no future real-F
coordinate is physically exposed across an S turn. The rule is totalized on
its selected cylinder, not claimed as a global recurring controller.

The explanations of what remains open are correspondingly evidential:
missing outer selectors, reverse legality, recurrence, terminal transfer,
and outcome arguments are listed as unproved duties. The interface map is not
offered as a causal account of nonexistence. This is precisely the discipline
required by rewritten section 68/section 73.

**Proposed repair:** none.

### Finding 7 — MINOR: the artifact's own provenance ledger omits and contradicts the campaign input/landing pair

> “Input/HEAD read during this round: `ab0fd965...`” and “No commit was
> created, so there is no Round-9 landed hash.”

**Independent recomputation.** The task-pinned input is
`9e57ea060462788841d1f8f761db894981b482e9`. The artifact lands in its
immediate child `e19e97f73f66154459500be1d578bb0a5a807592`; that commit adds
the Round-9 artifact, and the landed and current worktree blob are both
`9c999896f1e3f64aeb60bcbc3e4ffb16e181081c` (SHA-256
`d2b718a173601a8704c10a8b9929e1f94bb72b53198a3900402f1dc302b4c31d`).
Section 83 instead records an earlier authoring chronology, calls `9e57ea06`
a later non-evidentiary advance, and says the artifact was unlanded. Whatever
the local drafting history, that is not the required campaign provenance and
does not record the final unmodified landing.

The inherited Round5-Round8 corpus and the engine sources are unchanged
across the task input-to-landing edge, and the Round9 blob is exactly the
landed artifact audited here. The documentary error therefore supplies no
mathematical counterexample.

**Proposed repair:** replace section 83.1's Round-9 identity paragraph with
the input, landing, blob, and SHA-256 above; preserve its separate historical
Round-8 provenance note.

### Finding 8 — NOTE: the protected ingress pair satisfies every current `R_1` gate, including the transient debt checks

> “Both new real-S cells avoid the actual `W_*`. Theorem S69 therefore
> preserves `RES_1`; all other current entrance gates were just checked.”

**Independent recomputation.** After the nonterminal second F pair, both
boards are at S-role `FirstStone`; the real/shadow occupancies have sizes
nine/eleven, real S has four stones, `E_S={y_2}`, and `T(y_2)` is fresh and
legal. Select the actual fixed-order minimum real window `W_*`; it has deficit
one. The first ingress choice excludes current real occupancy, inverse shadow
occupancy, `W_*`, and every axis line through a pair of the four old real-S
stones. At most six such lines occur, and each meets a radius-eight ball in at
most seventeen cells, so the bound is

`9+11+6+6*17=128<217`.

The resulting `y_3` is fresh, legal, off `W_*`, and is not collinear with any
old pair. A deficit-one window through `y_3` would contain all five current
real-S stones, including the noncollinear subset `a,b,y_1`; therefore the
first append is first-safe. Before the associated certificate is installed,
the old singleton debt is still physical. Its inherited transient calculation
is

`delta'-m'=(delta-m)+1-1_{y_3 in W}>0`.

Thus the short two-debt microstate is shielded. `Shat@T(y_2)` is fresh and
legal. A win is disposition 2; a nonwin certifies `y_2`, leaves debt `y_3`,
and makes the next fresh image `T(y_3)` legal.

Among the four pre-`y_3` S stones there is at most one collinear triple:
two different triples among four points would share two points, hence lie on
the same engine line and make all four collinear. The line exclusions create
no triple involving `y_3`. For the one possible triple of span `s`, the union
of all length-six windows containing it has size `11-s<=9` for `2<=s<=5`,
and is empty for `s>5`. The second ingress exclusion therefore has size at
most

`10+12+6+9=37<217`.

The chosen `y_4` is fresh, supported, off `W_*`, and outside every window
containing the possible preceding triple. It cannot win: with only six total
S stones, a win would contain all six and would put `a,b,y_1` on one axis.
The analogous old-debt microstate is also safe: first-safety gave old
E-live deficit at least two, and the second append can lower it by only one
when no S placement remains. `Shat@T(y_3)` is legal; a win is disposition 2,
and a nonwin reaches common-live F `FirstStone`.

At that exit an urgent window through debt `y_4` would contain three of the
preceding stones. They would be the unique possible collinear triple, putting
`y_4` in the excluded danger union. Hence `U_E=empty`, `tau_E=0`, and the
current certificate `T(y_4)` is fresh and legal through `T(y_3)`. Avoidance
of `W_*` preserves scalar `RES_1` by S69. The just-completed S pair is causal,
first-safe, certificate-fresh/legal, service-admissible, nonterminal, and
common-phase. These are the complete current S59 entrance gates, not merely
the scalar inequality.

The full cadence is therefore:

| Segment after S15 | Real role | Shadow role | Nonterminal successor |
|---|---|---|---|
| line seed | `F@k_1,F@k_2` | `Fhat@z_1,Fhat@z_2` | S `FirstStone` |
| guarded pair | `S@y_1,S@y_2` | `Shat@f,Shat@T(y_1)` | F `FirstStone` |
| prepayment pair | `F@k_3,F@k_4` | `Fhat@z_3,Fhat@z_4` | S `FirstStone` |
| protected ingress | `S@y_3,S@y_4` | `Shat@T(y_2),Shat@T(y_3)` | F `FirstStone` |

**Proposed repair:** optional exposition only—insert the two inherited
transient-debt inequalities in S70's ingress proof. They close exactly and do
not change the theorem.

### Finding 9 — NOTE: S70's three dispositions are exhaustive at constructed scope, and its S50/S67 bypass is real but narrow

> “There is one totalized, prefix-causal ... continuation rule whose first
> interface disposition is exactly one of [cliff, `Shat` terminal, full
> entrance].”

**Independent recomputation.** Every event before `z_3` is nonterminal by
owner count. At `z_3`, the S58 predicate either holds or does not. In the
first case the rule reports disposition 1 at the common `SecondStone` prefix.
In the second case, the preceding window argument makes `z_4` nonterminal and
the rule reaches the protected ingress S pair. Each of its two associated
certificate appends either wins, giving the first physical disposition 2, or
is nonwinning. If both are nonwinning, Finding 8 proves every current entrance
gate and gives disposition 3. The temporal priority makes these alternatives
disjoint as first dispositions, and the construction proves there is no
unlisted legality, cadence, collision, service, or terminal exit on this
selected cylinder.

The bypass of S50's quantifier gap is substantive. S67 said only that *some*
shadow counterplay of a slow alleged winner survives through local placement
eight; it did not protect S50's liftable branch. S70 chooses a different legal
branch directly. If that branch would end at the sixth Fhat stone, the winning
window proves the already-reached `z_3` cliff. Otherwise the branch remains
live and reaches the full entrance. No surviving branch is relabeled after
the fact.

The bypass is nevertheless only an interface theorem. It works by accepting
S58 carrier failure as one requested disposition. It proves neither an
outcome after that cliff, arbitrary-real-S coverage, completion of the next
reserve cycle, nor recurrence. The artifact repeats all four limitations in
S70, sections 78.2-78.4, the obstacle ledgers, and caveats 40-42.

**Proposed repair:** none. Detached summaries must retain “constructed
selected history,” “next entrance,” and “cliff is carrier failure, not an
outcome.”

### Finding 10 — NOTE: S70.1 is a sound optional closure, not an automatic or universal consequence

> “If the two ingress S cells are additionally chosen off `W_0` ... the
> coupled trace therefore has a sound real-F closure.”

**Independent recomputation.** The additional exclusion is at most the six
cells of `W_0` in each already nonempty choice ball, changing the strict bounds
from `128<217` and `37<217` to at most `134<217` and `43<217`. The variant
therefore remains prefix-causal and legal. Initial S stones, the first guarded
pair, and both ingress cells then all avoid `W_0`; the four selected real-F
holes plus opener `x` occupy five of its cells. At the reached common-live F
`FirstStone`, its sixth cell is empty, within line distance at most five of
physical F support, and legal.

The actual next `sigma` prescription is queried at its engine phase. Under
section 53, that legal shadow append and the real unique-hole append form the
already-associated final paired event. The real append physically completes
six, so later debt service is vacuous and neither engine is continued. The
argument neither manufactures an extra move nor requires the shadow append to
have the same coordinate or geometry.

This is honestly optional: it uses a separately specified strengthening of
S70's selected ingress choices. Base S70 preserves the actual `W_*`, which
need not be `W_0`; it does not assert that every disposition-3 history already
has the corollary's seed protection. The corollary also says explicitly that
it is not an arbitrary-S response and does not refute `sigma`.

**Proposed repair:** none.

### Finding 11 — NOTE: S70.2 applies S60 correctly and defeats only the one-for-one carrier

> “For every `(sigma,h) in FAST_8^{S15}` ... the qualified S58 cliff is
> forced one microstep before S49 on this candidate-own history.”

**Independent recomputation.** Folded S60 turns `d_sigma(h)<=8` into the exact
value six. S70 supplies an actual legal Shat counterplay whose first six local
shadow placements have owners

`Fhat,Fhat,Shat,Shat,Fhat,Fhat`.

The first five cannot be an Fhat win by count. A certificate-created Shat win
would be a legal counterplay refuting the alleged-winning clause, so that
disposition is unavailable for a member of `FAST_8^{S15}`. If the line-seeded
shadow prefix were still nonterminal after local placement six, it could be
extended legally to a complete/maximal counterplay, contradicting the uniform
depth six. Hence `z_4` wins as Fhat's sixth stone. Its winning window contained
the four old Fhat stones before `z_3` and had exactly `z_3,z_4` as holes. Since
the real seed gives `mu_R=3`, `z_3` necessarily hit at `(mu_H,mu_R)=(2,3)`.

All values are actual prescriptions on S70's chosen history; the proof neither
substitutes S67's different surviving branch nor assumes fast-class
nonemptiness. The conclusion is exactly S58 failure of the query-first,
common-phase, one-for-one carrier. Lines 809-810, the theorem ledger, the P5
row, and caveat 41 all retain that this does not refute `sigma` or decide the
fast game outcome.

**Proposed repair:** none.

### Finding 12 — NOTE: S69 is an exact necessary-and-sufficient scalar `RES_1` test

> “The exit satisfies scalar `RES_1` exactly when there exists
> `W in A_{a'}(p)` with `W intersect Y=empty`.”

**Independent recomputation.** During the nonterminal S-role segment no real
F stone moves. Adding real-S stones removes exactly the old F-unblocked real
windows meeting `Y={y_1,y_2}`; every survivor retains its old F deficit.
Similarly, Shat stones remove shadow Fhat-unblocked windows they meet and
leave every surviving Fhat deficit unchanged. Therefore

`U_R^F(exit)={W in U_R^F(p):W intersect Y=empty}`

and `mu_H'>=mu_H`. Put `a'=1` when `mu_H'=1` and `a'=2` otherwise. Exit
liveness gives `mu_R'>=1`, so (74.1) is exactly `mu_R'<=a'`. By the displayed
identity for the exit real family, that is equivalent to an old real window
of deficit at most `a'` avoiding both S cells. This proves both directions of
(77.3), not merely the sufficiency of preserving `W_*`.

Representative entries recompute as follows:

| Entrance and shadow exit | Exact real condition at exit | Result |
|---|---|---|
| `mu_H=1,mu_R=1`; a shadow deficit-one window survives | some real deficit-one window avoids `Y` | necessary and sufficient |
| `mu_H=1,mu_R=1`; Shat blocks every shadow deficit-one window | some real deficit-at-most-two window avoids `Y` | threshold relaxes and may preserve `RES_1` |
| `mu_H>=2,mu_R<=2` | some real deficit-at-most-two window avoids `Y` | necessary and sufficient |
| `Y` hits `W_*` but an alternative adequate window survives | alternative window witnesses (77.3) | scalar survival, strict designated-reserve exit |
| `Y` hits every adequate window | exit real minimum exceeds `a'` | scalar break |

Thus avoidance of the designated minimum window is sufficient but not
necessary, exactly as items 3-4 claim.

**Proposed repair:** none.

### Finding 13 — NOTE: full `R_1` preservation really requires every S59 gate, but only by conditional class unfolding

> “These are exactly S59's S-pair admission clauses.”

**Independent recomputation.** S69.1 conditions on an already admitted
entrance and a reserve-handler-generated nonterminal F pair. That conditioning
discharges S59 cycle clauses 1-2. Its strict-preservation bullet then supplies
the remaining clauses 3-5:

- a causal, legal, first-safe, certificate-fresh, service-admissible,
  nonterminal real/shadow S pair;
- avoidance of the designated `W_*` by both real-S coordinates; and
- a common-live exit with `tau_E<=1`.

Section 77.3 expands the same requirements as gates G1-G5 and separately
classifies first-unsafe/terminal events, correct- versus wrong-role or illegal
certificate cases, second-placement/phase failures, designated-reserve loss,
and `tau_E=2` versus `tau_E>2`. It also gives terminal events priority and
states that failures can overlap. Consequently the assertion “every S59 gate
is required” is proved by unfolding the defined strict cycle; it is not an
unstated theorem that an outer strategy can force all gates.

The artifact is honest on that distinction: section 77.3 calls the sieve
definition-level and denies an outer forcing claim, while section 77.4 says
full `R_1` remains conditional. It also keeps dynamic quietness on the next F
pair orthogonal to this S-pair test and retains augmented `F-LOCK^+` rather
than canonical lock.

**Proposed repair:** optional clarity only—state in S69.1's proof that clauses
1-2 are discharged by its conditioning and the listed S-pair tests are clauses
3-5. A sound terminal closure is a stop, not live preservation.

### Finding 14 — NOTE: S70's line seed exists and its first F pair is legal, nonterminal, and cadence-correct

> “Choose the first real F-unblocked window `W_0` through `x` ... use the
> first two holes of `W_0` as noncanonical padding for the first post-S15
> real F pair.”

**Independent recomputation.** Exactly eighteen length-six windows contain
the real opener `x`: six shifts on each of three axes. A distinct S stone can
share such a window with `x` only when it is axis-aligned at distance
`d<=5`; it then blocks exactly `6-d<=5` of the six windows on that axis. The
two S15 real-S stones therefore block at most ten incident windows, leaving at
least eight choices for `W_0`.

At S15 the only real F stone is `x`, so the other five cells of an F-unblocked
`W_0` are physically empty. Each is within line distance at most five of
`x`, hence belongs to the radius-eight legal store. The first two selected
holes are distinct and sequentially legal. Pairing them with the actual
`z_1,z_2` gives the exact checkpoint ledger:

| Stage | Real `(F,S)` | Shadow `(Fhat,Shat)` | Next common role/phase |
|---|---:|---:|---|
| S15 | `(1,2)` | `(2,3)` | F `FirstStone` |
| after `z_1/k_1` | `(2,2)` | `(3,3)` | F `SecondStone` |
| after `z_2/k_2` | `(3,2)` | `(4,3)` | S `FirstStone` |

No append can win at those owner counts. Because all three real-F stones lie
in `W_0`, while real F owns only three stones total, the new real minimum is
exactly three. This prefix handler is noncanonical, as the artifact says; it
does not falsely enter `R_1` while `RES_1` is still impossible.

**Proposed repair:** none.

### Finding 15 — NOTE: the guarded S pair is physically legal and leaves singleton debt with `tau_E=0`

> “The resulting debt is `E_S={y_2}`, not empty ... Therefore
> `U_E=empty` and `tau_E=0`.”

**Independent recomputation.** At the first constructed S turn the real and
shadow occupancies have sizes five and seven. In the 217-cell ball `B_8(a)`,
the exclusions for current real occupancy, inverse shadow occupancy, `W_0`,
and the possible axis line through `a,b` have union size at most

`5+7+6+17=35<217`.

Thus `y_1` is fresh, supported, off `W_0`, has fresh image, and makes
`a,b,y_1` non-axis-collinear. Real S then owns only three stones, so this
first append is automatically first-safe. A fresh filler distinct from
`T(y_1)` exists among at least `217-7-1=209` cells and is only Shat's fourth
stone.

Prospectively adding the fixed legal certificate `T(y_1)` determines the
actual pure-strategy values `z_3,z_4`; `z_3` is only Fhat's fifth stone, so
`z_4` is defined. The second real choice excludes at most

`6+6+9+2=23<217`

cells: current real occupancy, `W_0`, inverse occupancy after the fixed
certificate, and the inverses of `z_3,z_4`. Actual chronology remains
`S@y_2` followed by `Shat@T(y_1)`; the prospective state was computation,
not an append. Both are legal and nonterminal, and `T(y_2)` remains fresh
through the reached F pair.

Afterward real S owns exactly four stones. Any E-urgent window through
`y_2` would have deficit at most two and would therefore contain all four,
forcing the already noncollinear subset `a,b,y_1` onto one engine axis.
That is impossible. Hence `U_E` is empty and `tau_E=0`, even though the
physical debt is the nonempty singleton `{y_2}`. Both real-S cells avoided
`W_0`, so the real minimum at `q_3` remains exactly three.

**Proposed repair:** none.

### Finding 16 — NOTE: S70's second-F-pair split is exhaustive, and every sixth-stone terminal is nested in the earlier cliff

> “If the pre-event `mu_H=2` and `z_3` hits a shadow deficit-two window ...
> This is disposition 1.”
>
> “The absence of disposition 1 forces `z_4` to be nonterminal.”

**Independent recomputation.** Before `z_3/k_3`, real F owns three stones,
all in `W_0`, so `mu_R=3`; shadow Fhat owns four, so `mu_H>=2`. The selected
`k_3` is a fresh supported hole of `W_0`. Both physical appends are
nonterminal by count and lead to the common F `SecondStone` query.

If `mu_H=2` and `z_3` meets a minimum shadow window, the exact pre-event state
is `(2,3)`, and folded S58 applies to every query-first, common-phase,
one-event-per-microstep, one-for-one selector. If that case is absent, then:

- an old minimum two survives when `z_3` misses every such window; or
- an old minimum at least three can fall by at most one.

Thus `mu_H>=2` at `q_4`. The fourth real F stone leaves `W_0` at deficit two,
and the total real-F count gives the matching lower bound, so `mu_R=2` and
`RES_1` holds before the genuine second query.

Suppose nevertheless that `z_4` wins. It is Fhat's sixth physical stone, so
all six Fhat stones belong to its winning window. At `q_3` that same unblocked
window contained the four old Fhat stones and had exactly the two holes
`z_3,z_4`. Therefore `mu_H=2` and `z_3` hit a deficit-two window. That is
precisely the already-prioritized S58 disposition. Consequently, on the
non-cliff branch, `z_4` is nonterminal. Pairing it with a distinct remaining
`W_0` hole leaves real F with five stones, all in `W_0`, so `mu_R=1` and
`RES_1` holds for every live shadow minimum.

This is an exhaustive constructed-interface split. It deliberately does not
convert the later S49 event into an outcome theorem.

**Proposed repair:** none.

## Per-theorem verdicts

| Result | Audit disposition | Exact boundary or required repair |
|---|---|---|
| S64 | **CONFIRMED** | Exact fast count/minimum census; no scalar reserve at any `q_j`; no outcome claim |
| S64.1 | **CONFIRMED** | Named S59 witness cannot be spliced into the same append-only one-for-one history |
| S65 | **CONFIRMED at named mechanism scope** | Reserve, two ordinary appends, section-53 closure, and terminal-moment S63 all fail; arbitrary outer carriers not excluded |
| S65.1 | **CONFIRMED** | Existing immediate real-F holes can be covered; later real outcome remains open |
| S66 | **CONFIRMED** | Earliest possible rolling entrance only; no claim that admission occurs |
| S67 | **CONFIRMED at shadow scope** | Slow depth supplies a live candidate-owned shadow branch, not its real lift |
| S67.1 | **CONFIRMED** | Exact opening-pair scalar sieve at folded S55-S58 scopes |
| S68 | **CONFIRMED as an interface partition** | Exhaustive for a fixed extension attempt; not synthesis or causal explanation |
| S69 | **CONFIRMED** | Necessary-and-sufficient scalar `RES_1` membership classification |
| S69.1 | **CONFIRMED conditionally** | Full strict preservation unfolds every S59 admission gate; no outer forcing theorem |
| S70 | **CONFIRMED at constructed next-entrance scope** | Exhaustive cliff/physical Shat stop/full current entrance on one selected first cycle |
| S70.1 | **CONFIRMED on the protected positive horn** | Optional policy variant fixed in advance; immediate physical real-F closure |
| S70.2 | **CONFIRMED for `FAST_8^{S15}`** | S60 forces the line-seeded S58 cliff; carrier failure only |
| Section 83 provenance | **MINOR ERRATUM** | Replace stale authoring chronology with input `9e57ea06`, landing `e19e97f7`, and landed blob identity |

## Named fast-conversion obstruction verdicts

| Proposed conversion | Audit disposition | Decisive recomputation |
|---|---|---|
| Same-history S59 named reserve | **CONFIRMED FAILS** | `RES_1` fails at every fast query and both eligible entrances; importing prepayment adds forbidden physical history |
| Two ordinary real-F appends | **CONFIRMED FAILS** | They raise the real-F count only from three to five |
| Section-53 paired final-event closure | **CONFIRMED FAILS** | It pairs existing appends but creates no sixth real stone and cannot continue a terminal engine |
| S63 conversion at the terminal microstep | **CONFIRMED FAILS** | The event is F-role and terminal; S63 requires a common-live mirror-clean first-unsafe S event |
| Any named conversion among the four | **NONE SOUND** | No counterexample to S65's named-mechanism obstruction was found |

## Overall verdict

**SOUND-WITH-MINOR-ERRATA.** No REFUTED or MAJOR finding survives
recomputation. The headline S70 controller, S70.1 closure, S70.2 fast-cliff
application, S69 classification, and all four negative conversion results
are sound at their expressly limited scopes. The fast game outcome and
`NL_F` remain **OPEN**. The only required correction is the Round-9
provenance ledger.

## Exact unresolved obstacles

1. **Full per-pair zero-lag coverage remains open.** S70 supplies one finite
   inherited-`T` prefix, not arbitrary-S recurrence, changing isometries,
   nonisometric or partial recodings, or indefinite one-repair service.
2. **Pre-checkpoint and recurring P3 remain open globally.** S70 completes
   the named first-cycle advance only; it does not repeat it through
   arbitrary later S pairs.
3. **Coverage outside strict `A_FS2` remains open.** Missing, blocked,
   wrong-role, illegal, unsupported, phase-lagged, unreflected-terminal, and
   high-transversal cases are avoided on S70's line, not handled generally.
4. **P5R and common-only real-win transfer remain open generally.** Every
   lag/recode still owes physical shielding, certification, F blocking, and
   same-step supply; section 53 adds no stone.
5. **Universal service and lock remain open.** S70's line handler is
   noncanonical; canonical `F-LOCK`, arbitrary service compatibility, and
   recurring portfolio admission are unproved beyond quiet/`R_1`
   `F-LOCK^+`.
6. **Universal shadow-F terminal fidelity and the fast outcome remain open.**
   S70.2 forces a cliff precursor and S64-S65 exclude the named repairs, but
   neither defeats `sigma` nor decides play after the carrier failure.
7. **Reverse legality for spatial carriers remains open.** Any inverse lift
   of S67's arbitrary survivor still owes current support, collision, S13,
   and S18 checks; S70 avoids rather than solves this duty.
8. **Global strategy domain and persistence remain open.** They hold on
   S70's finite append-only history, not on an arbitrary recurring outer
   construction.
9. **Global causality remains open.** S70's prefix choices are causal, but
   arbitrary backing, recoding, and repair still must avoid future-coordinate
   exposure across S turns.
10. **Universal window/certificate maintenance remains open.** One recurring
    handler must still combine legality, P2/P3, P5/P5R, common-only transfer,
    reassignment, and arbitrary S-created windows.
11. **High-transversal service and permanent fencing remain open.** The exact
    `tau_E=5` geometry, six-cell cost, availability, interruption, S
    occupation, reconciliation, and P3 compatibility have no construction.
12. **Strategy-specific global reachability and outcome remain open.** Fast
    class nonemptiness, post-cliff play, arbitrary-S coverage, completion of
    the next reserve cycle, indefinite recurrence, and an outcome theorem
    after S49 are all unproved.
