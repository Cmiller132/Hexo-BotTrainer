# Strategy stealing in engine Hexo, round 5: the total-exact two-escape ceiling, guarded lag, and window certificates

**Worktree:** `hunt/gap-raw` at input HEAD `67f996d1`  
**Date:** 2026-07-18  
**Global target:** `NL_F` remains **[OPEN]**.

This round obtains three scoped results. First, the support-cut argument
excludes `C_A^{K=2,total}`: the total-exact owner-faithful isometric subbranch
of (A) with a fixed lifetime budget of two coordinate-reactive episodes. S23
forces both episodes during the first tested S pair. Any earlier repair or
transfer failure already defeats that class; otherwise the candidate's own
next `sigma` pair yields an untransferred shadow-F terminal verdict or a third
support cut on the same legal, `sigma`-consistent history. This does not
exclude two episodes per S pair or other zero-lag window recodes.

Second, branch (B) has a genuine positive module. A physical lag queue admits
only shield-safe real-only S stones, retains every shadow filler physically,
and uses the canonical urgent-window transversal on the following F turn.
The deadline shield is preserved on the exactly named `A_FS2` coupled
segments, whose admissible adversarial S-coordinate projections are
first-safe and two-serviceable. The result discharges preventive
P5R and the real-board geometry component of item-4 service for that class.
P3-compatible realization remains conditional; this is not a global P3
construction.

Third, this round defines a window-faithful representation that has no global
cell map at all. It assigns physical shadow deficit windows to real window
obligations. A one-step transfer lemma proves P5R directly, and an explicit
pair of distinct overlapping real windows sharing one shadow certificate
proves that the representation family is genuinely outside the global
window-exact/isometric scope of S8.1.

All claims are hand proofs. No Cargo command, Lean build, harness, executable
search, or proof search was run. This authoring pass creates no commit.

## 36. Statement boundary and binding inherited contract

### 36.1 Target and status discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real second
player. As before,

`NL_F : exists pure sigma_F, for every pure sigma_S, S never wins`.

The clause `S never wins` permits a finite F win or an infinite nonterminal
history with neither six. Round-2 D2 remains **[PROVEN from the CITED Gale--Stewart open
determinacy theorem]** on the declared unbounded-board macro-game:

`NL_F  <=>  S has no winning strategy`.                         (D2)

Thus it is enough to refute every alleged winning S strategy, but this round
does not do so. Every named or load-bearing claim below is marked **PROVEN**,
**SKETCH**, **CONJECTURE**, **OPEN**, or **CITED**. Definitions are
stipulative. There are no machine-verified claims.

### 36.2 Production rules actually used [PROVEN]

The inherited rule model is unchanged. The initial state has `Player0` in
`Opening` (`packages/hexo_engine/rust/src/state.rs:149-160`). Forward play has
the nonterminal cadence

`F ; S,S ; F,F ; S,S ; ...`.

A normal placement is legal exactly when it is empty and belongs to the
physical radius-eight legal store
(`packages/hexo_engine/rust/src/rules.rs:11-44` and
`packages/hexo_engine/rust/src/legal.rs:17-18,123-145`). Board insertion and
the legal/window updates occur before the next placement is considered
(`packages/hexo_engine/rust/src/board.rs:83-105`). A six is checked after each
single placement and before phase advancement; a nonwinning `SecondStone`
passes control to the other owner at `FirstStone`
(`packages/hexo_engine/rust/src/state.rs:289-357`). The all-six predicate,
the three Q/R/QR axis-vector window families, and the eighteen incident
windows are at
`packages/hexo_engine/rust/src/tactics.rs:13-17,21-75,205-208,451-485`.
Terminal states expose no legal continuation
(`packages/hexo_engine/rust/src/state.rs:203-252`).

Physical histories are append-only under forward placement: successful
placements are pushed into history at `state.rs:265-273,302-307`. The public
apply/undo path is identified as the MCTS hot path at `state.rs:283-288`; as
required by the round-4 erratum, it is an analysis API outside the legal
forward `Placement` histories used here, not permission to erase, move, or
recolor a proxy or filler.

The executable carrier is `i16`
(`packages/hexo_engine/rust/src/coord.rs:9-16`); general results below have
the same `Z^2` scope as D2. Every explicit finite coordinate witness is
separately bounded well inside the carrier.

### 36.3 Corrected successor quantifiers and agenda labels

The folded round-4 section 35 is binding. Per observed real-S placement, a
candidate chooses (A) a zero-lag repair, (B) a shield-admissible lag/queue, or
(C) a same-step physical terminal certificate. Whichever branch it chooses,
it must also discharge service/reconciliation, persistent-stone accounting,
and the S18/S20/S12/S25 regressions. Checkpoint (28.1) is conditional on a
successful total-exact P3 transfer. A negative construction counts only when
it is selected on the candidate's own legal, `sigma`-consistent history.

For auditability, annotations `Agenda {i,...}` refer to the round-4 review's
authoritative ten-item open agenda. No annotation claims to discharge an item
beyond the theorem's displayed scope.

## 37. Total-exact subbranch (A): a fixed lifetime budget `K=2` fails

### 37.1 The exact candidate grammar

**Definition 37.1 (`G_A^2`, `Success_A`, and `C_A^{K=2,total}`).** Fix an
alleged winning pure shadow second-player strategy `sigma` and an S15 genuine
synchronization for that same `sigma`. Let `G_A^2` be the syntactic family of
deterministic causal total-exact branch-(A) update algorithms whose legal
partial executions obey the following rules whenever the algorithm acts:

1. the shadow is one genuine append-only Hexo history, every shadow-`Fhat`
   move is prescribed by `sigma`, and the real/shadow actors and phases agree
   at every common-live checkpoint;
2. before each real-S coordinate is selected, the candidate commits a total
   exact owner-faithful binding

   `O_H = T[O_R] disjoint-union P`, `T=t+g`, `g in D6`,

   at a normal common-live checkpoint with both represented and proxy parts
   nonempty;
3. every prescribed shadow-`Fhat` placement is paired with exactly one real-F
   placement, and a total exact binding is genuinely restored after the
   matched step unless a sound physical terminal certificate ends play;
4. when a real-S coordinate has an occupied committed target, its only
   nonterminal remedy is a zero-lag coordinate-reactive episode: append
   exactly one legal physical `Shat` stone and restore a same-phase total exact
   binding before the next engine placement;
5. arbitrary finite representation-only rebindings are allowed before
   commitment and while carrying out P3, but at most **two** S-coordinate-
   reactive episodes may occur over the entire continuation from
   synchronization; and
6. every terminal stop discharges the appropriate physical terminal
   direction. In particular, a proxy-assisted `Fhat` win is not called a real
   F win without a real physical certificate.

Lag, a queue, non-total or nonisometric representations, window-only
certificates, noninjective merges, unmatched extra physical placements, and a
budget reset on each S turn are outside this grammar. Put `Success_A(C)` when
`C in G_A^2` covers every coupled nonterminal successor against every legal
real-S continuation until a sound physical stop, fulfills all six rules, and
never truncates voluntarily. Finally define the extensional success class

`C_A^{K=2,total}={C in G_A^2 : Success_A(C)}`.

The grammar is nonempty **[PROVEN]**: fix the deterministic candidate
`C_0 in G_A^2` that follows the folded round-4 section-35 backing/filler
behavior on that named history, resolves every tie with the already-required
fixed coordinate and subset enumerations, and reports its first failed
promise on any history departing from that path. The round-4 legal partial
execution is then one legal partial execution *of `C_0`*, witnessing
nonemptiness of the algorithm class (a trace alone is not an algorithm —
review round-5 Finding 4). S32
proves that the extensional success class is empty. **Agenda {2,7,10}.**

### 37.2 The third-cut theorem

**Theorem S32 (fixed-total two-escape obstruction) [PROVEN].** For every
alleged winning `sigma`, every S15 synchronization for it, and every
corresponding `C in G_A^2`, `Success_A(C)` is false; hence the associated
`C_A^{K=2,total}` is empty. More precisely, for each such `C` there is a legal
real-S continuation on its own genuine `sigma`-consistent coupled history
such that `C` violates at least one Definition 37.1 promise earlier or, after
two successful episodes and a nonterminal second `sigma` pair, success
requires a third S-coordinate-reactive episode. **Agenda {1,2,7,10}.**

*Proof.* At S15 synchronization the physical counts are

`real (F,S)=(1,2)`, `shadow (Fhat,Shat)=(2,3)`.

Query `sigma` for its next sequential F pair. Neither shadow placement can
win: they give `Fhat` only its third and fourth stones. If a prescription or
its required real transfer/restoration fails, the first alternative of the
theorem already holds. Otherwise real F also has only three stones, and the
candidate reaches its own normal common-live S `FirstStone` checkpoint

`real (F,S)=(3,2)`, `shadow (Fhat,Shat)=(4,3)`.              (37.1)

Apply S22 to the candidate's actual committed binding. It supplies a
real-empty legal coordinate `c_1` whose committed shadow target is an occupied
proxy. Real S has only two stones, so `S@c_1` is nonwinning. Definition
37.1 therefore requires the first reactive episode. If its legal append or
restoration fails, a Definition 37.1 promise has already failed. Otherwise
the successful episode has one physical `Shat` append and restores the exact binding at common
`SecondStone`; the counts are `(3,3)` and `(4,4)`. Reapply S22 to that newly
committed binding. Its selected `c_2` is fresh, legal at `SecondStone`, and
gives real S only four stones; its associated shadow append can give `Shat`
only five. Thus neither side can terminate, and the second occupied target
forces the second episode. Failure inside it is again an earlier promise
failure; successful restoration completes the ordered S turn and gives

`real (F,S)=(3,4)`, `shadow (Fhat,Shat)=(4,5)`,              (37.2)

with the F role at `FirstStone`. The transition to F is the nonwinning
`SecondStone` branch at `state.rs:330-333`.

Now follow the candidate across `sigma`'s next genuine F pair. The first
matched placement gives real F four stones and `Fhat` five, so neither can
win. If either matched placement or an exact restoration fails, P3 fails. On
the second placement, real F has only five stones and cannot own a six. If
`Fhat` wins with its sixth physical stone, the verdict cannot transfer:
shadow `Fhat` is terminal with six physical stones, while real F has only five
and cannot have any physical six certificate. The shadow has no continuation.
This is exactly an unresolved S20-type P5 failure; no post-terminal restored
binding is assumed.

It remains to consider the nonwinning second prescription. The candidate's
own history then reaches

`real (F,S)=(5,4)`, `shadow (Fhat,Shat)=(6,5)`,              (37.3)

at S/`Shat FirstStone`. Representation-only rebindings do not change these
physical owner counts. Total injective owner fidelity therefore leaves
exactly one current proxy of each shadow role, irrespective of which old
stones the candidate rebinds. Commit the candidate's actual post-P3 binding
and apply S22 again. It selects a real-empty legal `c_3` with an occupied
target. Real S has four stones, so `S@c_3` is nonwinning.

There is also no hidden same-step `Shat` terminal escape under the theorem's
assumption that `sigma` is winning. If a legal `Shat@w` won from the current
genuine prefix, define a total pure `Shat` strategy that repeats the finitely
many displayed on-path choices, plays `w` there, and uses the least legal
coordinate in a fixed enumeration at every other decision node. Round-2 D1's
non-dead-end argument supplies off-path moves, including `SecondStone` nodes.
The play against `sigma` would reach `Shat@w` and defeat `sigma`, a
contradiction. Hence the occupied target requires a third zero-lag episode or
a switch to a branch outside Definition 37.1. Both contradict the candidate's
promise. QED.

Every adversarial coordinate in the proof is recomputed from the candidate's
successive committed binding. The intervening shadow moves are `sigma`'s
actual prescriptions. Thus S32 meets the corrected reachability quantifier;
it is not an S30-style labeled-state gadget.

### 37.3 Sharp boundary of S32 [PROVEN]

S32 refutes a fixed **total** budget of two. It does not refute two reactive
episodes **per S pair**. After nonwinning `c_2`, the engine passes to F, so a
third cut made immediately then would identify a legal coordinate but would
not give S a coordinate it is entitled to choose. The proof must, and does,
cross the next actual `sigma` pair and split on its possible second-placement
terminal verdict before recomputing the third cut. Branch (B), branch (C),
non-total maps, nonisometric maps, and window certificates remain outside
S32. **Agenda {2,10}; per-pair `K=2` remains OPEN.**

## 38. Branch (B): guarded lag with exact two-stone service

### 38.1 Physical queue semantics

**Definition 38.1 (rolling one-cell physical lag queue).** Fix a
translation/D6 map `T` used only for the S-role certificate on one genuine
coupled history. At every live checkpoint choose

`C_S subseteq X_S`, `E_S=X_S\C_S`,

where `c in C_S` only if the physical shadow board actually contains
`Shat@T(c)`. The rolling state requires `E_S=empty` or `E_S={e}`.

After observing a legal real `S@y`, the branch-(B) append is determined as
follows.

- If `E_S={e}`, require `T(e)` to be fresh and legal on the physical shadow
  board, append `Shat@T(e)`, physically reconcile `e`, and leave the new real
  stone `y` as the sole debt: `E'_S={y}`.
- If `E_S=empty`, append the first legal shadow filler `f != T(y)` under a
  fixed enumeration and put `E'_S={y}`. If no such filler is available, this
  branch is unavailable.

Thus every queued cell carries the deadline shield until the next S-role
append physically certifies it; the last cell of an ordered pair carries its
debt across the F turn. The first filler and every old proxy remain on the
append-only shadow board forever, even after their bookkeeping role changes.
They continue to affect legality, blocking, and terminal windows. A label
change is not reconciliation.

The queue is representation lag, not engine-phase lag: one physical S-role
stone is appended to each game on the coupled step. Any branch-(B) `Shat`
append — reconciliation `Shat@T(e)` *or* an enumerated filler — ends the
physical module when it completes a shadow six; it contradicts alleged-winning
`sigma` only when the outer P3 carrier has kept the prefix `sigma`-consistent.
Otherwise the histories remain common-live. This module authorizes neither an
extra physical shadow move nor undo. **Agenda {3,8}.**

For the real board retain round 4's definitions. A real window `W` is E-live
when `W intersect E_S != empty` and `W intersect X_F=empty`, and

`delta(W)=6-|W intersect X_S|`.

At S `FirstStone`, S `SecondStone`, and throughout F's intervening turn put
`m=2,1,0`, respectively. The deadline shield is

`delta(W)>m for every E-live W`.                            (38.1)

At an F `FirstStone` checkpoint let `U_E` contain the E-live windows with
`delta<=2`, let `H_W=W\X_S`, and let `tau_E` be the minimum size of a set
hitting every `H_W`, exactly as in S29. Fix enumerations of finite coordinates
and finite subsets; when `tau_E<=2`, let `K_E` be the first minimum
transversal. This makes the service choice causal and deterministic.

### 38.2 The named continuation class

**Definition 38.2 (`A_FS2`, rolling first-safe/two-serviceable
continuations).** Relative to the deterministic handler in Definition 38.1,
let `A_FS2` be the class of finite physical coupled trace segments generated
by the handler and satisfying, recursively on each segment's own prefix:

1. every required old-debt certificate `T(e)` is fresh and shadow-legal, or
   its append is an immediate physical `Shat` terminal stop; likewise any
   enumerated filler append that completes a shadow six is an immediate
   physical `Shat` terminal stop of the module;
2. after every lagged `FirstStone y`, each F-unblocked six-window through `y`
   contains at most four real S stones in the post-state, equivalently
   `delta>=2`;
3. every lagged `SecondStone y` is nonwinning;
4. after every completed S pair, the resulting urgent family has
   `tau_E<=2`; and
5. the physical F-service operator keeps `C_S,E_S` fixed, places the members
   of `K_E` and then least-legal padding if necessary, or stops on an earlier
   real F win; and
6. unless real F has stopped the trace, the shadow appends a genuine legal
   nonwinning `Fhat` pair so that the next rolling step is again at a common
   S-role phase. This pair need not be prescribed by `sigma` and is not yet
   identified with the real service pair by a P3 theorem.

This is a recursively defined continuation class, not an arbitrary labeled
state. Clauses 2--4 exactly name its admissible adversarial real-S coordinate
projections; clauses 1, 5, and 6 are handler-side conditions. It does **not**
assume that an arbitrary `sigma` pair maps to the service cells. Composition
with P3 is stated separately after S34. Lemma S35 exhibits a reachable prefix
with actual reconciliation and active `tau_E=1` service, so the scope is not
the empty-queue case. **Agenda {2,3,4,8,9}.**

### 38.3 Admission and service algebra

**Lemma S33 (phase-exact rolling admission) [PROVEN].** Assume the deadline
shield before a real-S placement and that Definition 38.1 physically
reconciles the old singleton debt, if any.

1. A new first-coordinate lag is shield-admissible exactly when every newly
   E-live window has post-placement `delta>=2`.
2. Every nonwinning second-coordinate lag is shield-admissible.
3. At the following F turn, if `tau_E<=2`, a nonwinning real F pair containing
   `K_E` restores `delta>2` before the next S `FirstStone`; if a service
   placement wins for F, that is a sound earlier stop.

**Agenda {3,4}.**

*Proof.* Physical reconciliation removes every old E-live obligation. The
only newly E-live windows are those through the new singleton debt. After an
S first placement, one S placement remains, so the post-placement value is
`m=1`; strict shielding is precisely `delta>1`, equivalently integer
`delta>=2`, on each such window.

After an S second placement the post-placement value is `m=0`. If a newly
E-live window through the new real-only stone had `delta=0`, that physical
placement would have completed a real S six. The nonwinning premise excludes
this, so every such window has `delta>=1>0`.

For item 3, hold `C_S,E_S` fixed. The coordinates of `K_E` hit the physical
holes of every urgent window. Each is legal because a one- or two-hole
six-window already contains four or five S stones, putting each hole within
distance at most two of physical support. If fewer than two service cells are
needed, use the fixed enumeration to choose fresh legal padding. Unless F
wins earlier, the completed pair permanently blocks every urgent window;
every unblocked E-live window then has `delta>=3>2`. This is S29's
sufficiency argument with the deterministic service selector made explicit.
QED.

### 38.4 Conditional branch-(B) invariant

**Theorem S34 (rolling guarded-lag/service theorem) [PROVEN at `A_FS2`
scope].** Start from any genuine shielded rolling state. Against every finite
continuation in `A_FS2`, the physical queue handler maintains the deadline
shield after every live real-S placement. If the following Definition
38.2(5) service pair completes nonwinning, it restores `delta>2` before the next S
`FirstStone`. Consequently no real S win on a handled branch-(B) step can
meet `E_S`; a terminal coordinate cannot be admitted to this branch.
**Agenda {2,3,4,8,9}.**

*Proof.* Induct over coupled single placements. Consider first the transient
physical state between the real append `S@y` and the shadow append
`Shat@T(e)`, in which the old debt `e` is not yet reconciled and `y` is also
unmatched. For every old E-live window `W` through `e`, the real append and
the role-count change give

`delta'(W) - m' = (delta(W) - 1_{y in W}) - (m - 1) = (delta(W) - m) + 1 - 1_{y in W} > 0,`

so no old-debt window can become terminal in the transient real state; every
new window through `y` is instead handled by the first-safe or nonwinning
admission test below. The shadow append then physically
certifies the old singleton debt, so its E-live family disappears. Lemma
S33(1) admits exactly the first-safe new debt after `FirstStone`; Lemma S33(2)
admits every nonterminal new debt after `SecondStone`. Thus the shield holds
with `m=1` and then `m=0`.

At the F checkpoint, `A_FS2` supplies `tau_E<=2`. The deterministic service
operator places a pair containing `K_E` while keeping the semantic sets
fixed. Lemma S33(3) restores `delta>2` before the next S turn, unless real F
has already won. Definition 38.2(6) supplies a genuine nonwinning shadow pair
and restores the common S-role phase for the next rolling step; it is not used
as a P3 theorem. This closes the induction.

If a real S terminal window met `E_S`, it would have `delta=0`, contradicting
the maintained strict inequality even on the terminal append. Therefore a
terminal coordinate cannot be the new rolling debt. QED.

S34 is preventive P5R for admitted traces. A first-unsafe or terminal real-S
coordinate makes branch (B) unavailable; a global candidate must handle that
coordinate with branch (A) or a same-step physical branch-(C) certificate.
S34 does not construct that response.

S34 is the requested positive branch-(B) result at an honest scope. It is
stronger than merely naming `C_shield`: it gives actual next-step
reconciliation, the exact first/second admission rule, a deterministic
service operator, and the recursively named continuation class on which the
operator closes.

**P3 composition boundary [OPEN].** S34 composes with a stealing
coupling only if a genuine P3 carrier realizes `sigma`'s sequential `Fhat`
prescriptions as the selected real service pair and supplies a same-step
real-F terminal certificate for every shadow-`Fhat` terminal prescription.
A nonisometric temporal pairing is enough for nonterminal phase/legality, but
does not itself prove future representation or terminal fidelity. Therefore
S18 remains a reverse-legality regression and S20 remains the exact P5
obstruction. **Agenda {1,6,7}: OPEN beyond this conditional interface.**

### 38.5 Active-service witness

**Lemma S35 (`A_FS2` has a reachable reconciliation-and-service prefix)
[PROVEN].** Put `T(q,r)=(q+2,r)` and choose the fixed enumeration so that the
first eligible filler is `(0,1)`, singleton `{(0,5)}` precedes singleton
`{(0,6)}` among the minimum transversals, and `(1,5)` is the first eligible
padding coordinate at the service state. Start with the S15 prefix and the
following successfully transferred F pair:

```text
real:
F@(0,0); S@(0,1),S@(0,2); F@(1,0),F@(2,0)

shadow:
Shat@(0,0); Fhat@(1,0),Fhat@(2,0);
Shat@(2,1),Shat@(2,2); Fhat@(3,0),Fhat@(4,0).
```

Now execute the rolling branch twice:

```text
real:   S@e=(0,3), then S@y=(0,4)
shadow: Shat@f=(0,1), then Shat@T(e)=(2,3).
```

The first event creates `E_S={e}` with `delta>=3>1`. The second physically
reconciles `e`, leaves `E_S={y}`, and is nonwinning. At the resulting F
checkpoint the only urgent E-live window is

`W={(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)}`,

with `H_W={(0,5),(0,6)}`. Hence `tau_E=1`; take the fixed order to select
`(0,5)` and pad with `(1,5)`. The real service pair `(0,5),(1,5)` and shadow `Fhat` pair
`(2,5),(3,5)=T[(0,5),(1,5)]` are legal and nonwinning. The real blocker
`(0,5)` restores `delta>2` for `E_S={(0,4)}`. S35 is a P3-shaped,
phase/legality-paired physical trace: its shadow pair is legal and
coordinatewise associated with the real service pair, and its on-branch
choices extend to *some* legal pure strategy — not claimed winning, so this
is no advance for arbitrary alleged-winning `sigma`. **Agenda: geometry side
of {3,4} and physical persistence in {8}; item 1 is advanced only by S32's
negative crossing — arbitrary positive P3 composition remains OPEN;
Agenda 10 is not advanced beyond reachable-prefix nonvacuity.**

*Proof.* Every coordinate is fresh. Each real vertical S placement is
adjacent to its predecessor; the initial shadow S images use `T`; filler
`(0,1)` is adjacent to the shadow opening; and `T(e)` is adjacent to the prior
shadow images. Before service, real S owns only `(0,1),...,(0,4)`. Of the six
R-axis windows through `(0,4)`, the only unblocked deficit-two window is
`W`: the two other intervals containing all four S stones contain real
`F@(0,0)`, while every remaining interval has deficit at least three. No Q or
QR window through `(0,4)` contains four S stones. This proves `tau_E=1`.

The service cell `(0,5)` is adjacent to real S support and `(1,5)` is adjacent
to it. Their shadow images are supported by the physical shadow S chain and
then by one another. Before service, shadow `Fhat` has only the q-axis run
`(1,0),...,(4,0)`; the separate r=5 pair creates no six. Real F has only five
stones after service. Thus every displayed prefix is nonterminal until the
claimed endpoint. All coordinates have hex norm at most eight and every
radius-eight halo at most sixteen, safely inside `i16`. QED.

The next rolling S pair can also reconcile `(0,4)` and then `(1,4)` by playing
real `(1,4),(2,4)` against shadow `(2,4),(3,4)`, leaving only `(2,4)` queued;
the first step is first-safe and the pair is nonterminal. This confirms actual
queue rotation rather than static labeling **[PROVEN for this extension;
Agenda {3,8}]**.

The displayed `Fhat` choices extend to a legal pure strategy off the branch,
but that strategy is not claimed winning. S35 establishes mechanism and
geometric nonvacuity only; S34 is universal exactly over finite `A_FS2`
continuations.

## 39. Window-faithful non-isometric representations

### 39.1 Physical deficit certificates

Let `W_6` be the engine six-window family. At a genuine coupled checkpoint,
let `E_S` be the outer coupling's physically honest set of unreconciled real-S
stones. Define

`U_E={W in W_6 : W intersect E_S != empty, W intersect X_F=empty}`,

`U_H={V in W_6 : V intersect X_Fhat=empty}`,

`delta_R(W)=6-|W intersect X_S|`,

`delta_H(V)=6-|V intersect X_Shat|`.                        (39.1)

For `V in U_H`, every cell not counted as `Shat` is physically empty.

**Definition 39.1 (window-faithful deficit representation).** A physical
window-deficit certificate is a history-causal selector

`nu: U_E -> U_H`                                            (39.2)

such that

`delta_H(nu(W)) <= delta_R(W)` for every W in U_E.           (39.3)

The selector certifies obligations, not stones. It defines no coordinate map,
not even on occupied cells; different real windows may use the same shadow
window. Every count in (39.1) is an actual engine occupancy count. Changing
`nu` changes no board. A complete-trace handler checks (39.3) after every
associated physical S/`Shat` append, including an append on which the real
engine terminates. A newly unmatched real-S cell enters `E_S` before that
post-append test, and no cell leaves `E_S` through labels alone. Fix a global
enumeration of lattice windows for every selector tie-break. **Agenda
{3,8}.**

For comparison only, an injective point map `f:H->H` **point-induces** `nu`
on `U_E` when `f[W]=nu(W)` for every `W in U_E`. Definition 39.1 does not
require such an `f`.

This is a P5R representation module only. It makes no assertion about
nonterminal move legality, `sigma`'s `Fhat` prescriptions, common-only real
wins, P3, or cadence restoration.

### 39.2 A reachable representation not induced by any injective point map

**Lemma S36 (genuinely non-stonewise certificate) [PROVEN].** Take the exact
round-4 section-30.3 witness and append the first F/`Fhat` pair from its folded
terminating extension. The physical prefixes are

```text
real:
F@(0,0); S@(1,1),S@(2,1); F@(3,0),F@(4,0);
S@(0,5),S@(1,5); F@(-1,0),F@(-2,0)

shadow:
Shat@(0,0); Fhat@(-1,0),Fhat@(-2,0);
Shat@(-1,1),Shat@(0,1); Fhat@(-3,0),Fhat@(-4,0);
Shat@(0,2),Shat@(-1,5); Fhat@(3,0),Fhat@(4,0).
```

They are legal, nonterminal, and end at S/`Shat FirstStone`. Put
`E_S={(0,5)}`. Every E-live real window has `delta_R>=4`. On the shadow board
let

`V={(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)}`.

It is `Fhat`-unblocked and its first three cells are physical `Shat` stones,
so `delta_H(V)=3`. The constant selector `nu(W)=V` is therefore a window-
deficit certificate for every E-live real window.

This assignment cannot be induced by any injective point map. In particular,
the distinct E-live windows

`W_Q={(0,5),(1,5),(2,5),(3,5),(4,5),(5,5)}`

and

`W_R={(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)}`

both receive `V`. If an injection `f` induced the assignment, then
`f[W_Q]=f[W_R]=V`, and injectivity would imply `W_Q=W_R`, false. **Agenda
{3}; the witness is reachable but is not claimed forced by a winning
`sigma`, so Agenda 10 remains open.**

*Proof.* The inherited round-4 proof and folded extension establish the two
physical histories. Through `(0,5)`, the only axis companion among the real S
stones is `(1,5)`, so every incident real window contains at most two S
stones. None of the displayed `Fhat` stones lies in `V`, and precisely
`(0,0),(0,1),(0,2)` in `V` are `Shat`-owned. The deficit inequalities and the
injectivity contradiction follow. QED.

Many-to-one **window assignment** is not a many-to-one stone merge inside
S22. No physical stone represents two simultaneous real stones, and S22's
total exact point-image premise is absent. Round-2 S8.1 is likewise
inapplicable because there is no global window-exact injection.

### 39.3 Exact one-append maintenance

**Lemma S37 (fixed-selector maintenance formula) [PROVEN].** Hold `E_S` and
`nu` fixed through one legal paired append

`real S@y / shadow Shat@z`,

absent an earlier sound terminal stop. For every old E-live `W`, put

`s(W)=delta_R(W)-delta_H(nu(W)) >= 0`.

Then

`s'(W)=s(W)-1_{y in W}+1_{z in nu(W)}`.                    (39.4)

Consequently the fixed certificate survives exactly when every tight
affected window (`s(W)=0` and `y in W`) has `z` in the current physical hole
set `nu(W)\X_Shat`. One shadow append services all tight affected
certificates exactly when the intersection of those hole sets contains the
chosen legal `z`; an empty family imposes no restriction. If the intersection
is empty, no one-append update preserving this fixed `nu` exists. **Agenda
{2,3}.**

*Proof.* The real append reduces `delta_R(W)` by one exactly when `y in W`.
The legal shadow append is physically empty beforehand, so it reduces
`delta_H(nu(W))` by one exactly when `z in nu(W)`. Subtraction gives (39.4).
For non-tight windows one unit of slack absorbs an unmatched real decrement;
for tight affected windows the shadow decrement is necessary and sufficient.
The intersection statement is the simultaneous version of the same
condition. QED.

Dynamic reselection of `nu` is outside the final obstruction sentence. S37
is an exact local service test, not a universal update algorithm. An update
that newly admits `y` to `E_S` is outside S37's fixed-`E_S` premise and must
separately certify the newly incident windows.

### 39.4 P5R transfer and the first obstruction

**Theorem S38 (P5R from physical deficit cover) [PROVEN at conditional
complete-trace scope].** Suppose a genuine legal coupled trace, with no
earlier unjustified terminal state, maintains Definition 39.1 after every
associated S/`Shat` append, including a possibly real-terminal append. Then
every real S win whose window meets current `E_S` produces a physical
shadow-`Shat` win no later on that coupled step. **Agenda {3}.**

*Proof.* Let `W` be such a terminal real window. It has no F stone and
`delta_R(W)=0`, so `W in U_E`. Post-append certification gives a physical
`Fhat`-unblocked shadow window `V=nu(W)` with

`0 <= delta_H(V) <= delta_R(W)=0`.

All six cells of `V` are therefore actual `Shat` stones. The physical all-six
predicate and win-before-phase-advancement rule make the shadow terminal for
`Shat` on or before the associated append. No stonewise image of `W` is used.
QED.

**Lemma S39 (winning-strategy deadline barrier) [PROVEN].** Let a genuine
legal nonterminal shadow prefix be consistent with an alleged winning
`Fhat` strategy `sigma`, with `Shat` to act at `FirstStone` (`m=2`) or
`SecondStone` (`m=1`). Every `Fhat`-unblocked shadow window satisfies

`delta_H(V)>m`.                                             (39.5)

**Agenda {3,8,10}.**

*Proof.* Otherwise let `1<=k=delta_H(V)<=m`; positivity follows from
nonterminality. At `FirstStone`, `k` is one or two; at `SecondStone`, it is
one. The window already contains at least four `Shat` stones, and every hole
is at line distance at most five from such physical support, hence is legal
under radius eight. `Shat` fills the holes in the current turn. A first-
placement win already defeats `sigma`; otherwise the sequential update makes
the second hole legal and filling it completes `V`. Extending these finitely
specified choices by least-legal moves off the branch gives a total pure
counterstrategy, contradicting that `sigma` wins against every `Shat`
strategy. QED.

**Corollary S39.1 (deficit certificates cannot beat the deadline shield)
[PROVEN].** On any ongoing common-phase history consistent with an alleged
winning `sigma`, a window-deficit certificate implies

`delta_R(W) >= delta_H(nu(W)) > m`

for every E-live `W`. Thus this natural non-stonewise, no-total-point window
representation cannot admit a shield-unsafe lag state. Either the certificate
does not exist on the candidate's own history, or the physical shadow already
contains the finite `Shat` counterplay that refutes `sigma`. **Agenda {3,8,
10}.**

S39.1 is a first obstruction, not a collapse theorem for all window
representations. Phase-lagged event certificates, multi-step certificates,
dynamic selector changes, and earlier physical terminal stops remain outside
it.

## 40. Full obligation ledger after round 5

### 40.1 Authoritative ten-item agenda

| Review item | Round-5 status | Exact advance and remaining gap |
|---:|---|---|
| 1. Pre-checkpoint P3 transfer | **OPEN globally; PARTIAL at scope** | S32 crosses two actual `sigma` pairs and treats transfer/restoration failure as a genuine `C_A^{K=2,total}` failure. S35 gives one legal P3-shaped, phase/legality-paired service trace (not arbitrary-winning-`sigma` compatible). Composing S34 still requires an outer carrier for arbitrary `sigma`; checkpoint (28.1) remains conditional. |
| 2. P2/P4 at each real-S coordinate | **PROVEN negative for `C_A^{K=2,total}`; PROVEN positive on `A_FS2`; OPEN globally** | S32 forces a third occupied target after the lifetime budget is exhausted. S33/S34 give a legal rolling branch-B rule on first-safe continuations. Two repairs per S pair, other zero-lag recodes, general lag, and dynamic window recoding remain open. Round-2 S13 (fixed-isometry one-stone FIFO frontier failure) remains a binding regression for any FIFO interpretation of lag; the rolling queue differs from it by persistent genuine fillers/proxies and physical reconciliation. |
| 3. P5R during every lag/recode | **PROVEN for two conditional modules; OPEN globally** | S34 proves the deadline shield for the rolling queue on `A_FS2`. S38 transfers E-meeting wins for complete traces maintaining physical deficit certificates. S39.1 proves that this natural window representation cannot certify shield-unsafe debt on a winning-`sigma` history. Round-2 S14 (literal one-S-stone-lag terminal-count failure) remains binding for any unguarded literal lag; S34 avoids it only on shield-admitted traces. |
| 4. F-service compatibility | **PROVEN as real-board service on `A_FS2`; P3 composition OPEN** | The canonical `K_E` rule realizes S29 whenever `tau_E<=2`; S35 exercises it. No theorem makes every arbitrary `sigma` pair realize `K_E`, and `tau_E>2` states remain possible. |
| 5. Permanent-fence installation | **OPEN; no new advance** | S31's six-blocker geometry remains binding. Availability, interrupted installation, and compatibility with prescribed F moves are unsolved. |
| 6. Reverse shadow-to-real legality | **OPEN** | A temporal P3 carrier may pair a legal shadow prescription with a legal service cell, but no universal carrier is constructed. S18 and the sequentially updated unsupported/collision sets remain mandatory, and S13 remains binding on its fixed-isometry FIFO schedule. |
| 7. Shadow-`Fhat` terminal fidelity | **OPEN; sharpened negative use** | S32 shows that the intervening sixth `Fhat` stone, if terminal, is necessarily untransferable at counts six versus five. S20 is not repaired; S34's composition corollary makes same-step physical P5 a premise. |
| 8. Strategy domain and physical persistence | **PARTIAL at module scope; OPEN globally** | Definitions 38.1 and 39.1 retain every filler/proxy, and each displayed S-role append is physical and legal. Branch B does not prove that its F-service partner is prescribed by the fixed alleged-winning `sigma`; the P0/P3 interface and arbitrary future recodes still owe one `sigma`-consistent history and phase relation. |
| 9. Causality | **PROVEN for the local selectors; OPEN globally** | S32 commits before each adversarial coordinate. Branch-B debt/service choices use only the observed prefix; `K_E` is selected after the S pair, so it does not preannounce an F coordinate across that S turn. An outer P3 carrier may still trigger S12 and is not supplied. |
| 10. Strategy-specific reachability and outcome | **PROVEN for the S32 obstruction; OPEN for the global outcome** | `c_1,c_2,c_3` are selected from the candidate's successive bindings on its own `sigma`-consistent history. S35 and S36 show only reachable mechanism states, not forced states for a winning `sigma`. No arbitrary winning strategy is refuted; `NL_F` stays open. |

### 40.2 P0--P6/P5R cross-ledger

| Obligation | Status after round 5 | Binding disposition |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **PROVEN at S15; PARTIAL for the new modules; OPEN globally** | S32 uses only actual `sigma` prescriptions. Branch B proves legal physical reconciliation/filler appends but not that its service-side F pair is the fixed `sigma` pair. Window certificates alone say nothing about P0. |
| `P1 OPENING/CADENCE` | **PROVEN for cadence/legal-prefix component** | S15 remains the start. Rolling branch B appends one S-role stone on each side and therefore preserves phase locally; arbitrary P3 restoration remains conditional. |
| `P2 REAL->SHADOW` | **PARTIAL** | `C_A^{K=2,total}`, the total-exact isometric subbranch, is excluded. `A_FS2` supplies a genuine rolling append when its freshness/legality condition holds. S37 gives the exact one-append condition for fixed window selectors. |
| `P3 SHADOW->REAL` | **OPEN** | S34 supplies the desired service pair, not a universal mapping of `sigma` to it. S18 remains live. |
| `P4 COLLISION` | **PARTIAL** | S32 strengthens the cut obstruction for `C_A^{K=2,total}` to lifetime budget two. Branch B avoids exact occupied targets by physical debt rotation on `A_FS2`. |
| `P5 SHADOW-F-TERMINAL` | **OPEN** | S32's six-versus-five fork and inherited S20 show why a physical certificate is indispensable. |
| `P5R REAL-S-TERMINAL-REFLECTION` | **PROVEN in `C_shield`, `A_FS2`, and deficit-certified traces; OPEN globally** | S34 is preventive; S38 is a non-stonewise physical certificate route. S14 and S25 remain mandatory terminal-memory regressions for any unguarded literal lag. |
| `P6 CAUSALITY` | **PARTIAL** | The new selectors are prefix-causal, but no universal P3 carrier is shown safe from S12. |

## 41. Hostile-review attack surface

The following limitations are part of the claims, not optional caveats.

1. **S32's budget is total.** It does not refute two episodes per S pair. The
   third cut is valid only after crossing `sigma`'s actual next F pair and
   recomputing the candidate's binding.
2. **The F-terminal fork in S32 is load-bearing.** If `Fhat` wins on its sixth
   stone, real F has five and cannot win. If it does not win, only then is the
   third common-live cut available. Continuing through a terminal shadow would
   violate `rules.rs:11-14` and `state.rs:203-252`.
3. **No abstract negative gadget is used.** All S32 coordinates are functions
   of the candidate's committed data. S35, S36, S25, and S30 remain positive
   or diagnostic witnesses unless a later theorem proves winning-`sigma`
   reachability.
4. **`A_FS2` is conditional but not vacuous.** S35 proves genuine finite
   reconciliation and service. It does not prove a complete winning-`sigma`
   trace, so S34 is stated as an arbitrary finite-horizon invariant theorem,
   not as a nonempty global success class.
5. **First-safety is mandatory.** The S25 divergence has `delta=1=m` after
   its first coordinate and is rejected. A global candidate must use branch
   (A) or (C) there. S30 and any other `tau_E>2` service state lie outside
   `A_FS2`; no unreachability claim is made.
6. **The service pair is not yet P3.** Choosing `K_E` after the S pair avoids
   preannouncing those service cells across that turn, but mapping arbitrary
   `sigma` prescriptions to them while preserving future legality and terminal
   meaning is open. Nonterminal temporal pairing does not solve S20.
7. **Every old physical stone persists.** Queue rotation changes semantic
   debt only. The first filler and all proxies still occupy cells, supply
   radius-eight growth, block the other owner, and enter all terminal windows.
8. **Definition 39.1 requires `Fhat`-unblocked shadow windows.** Otherwise
   `delta_H` would not count physical holes and S37--S39 would be false. The
   certificate is checked after a possibly terminal append; stored engine
   phase need not advance on that step.
9. **S37 fixes `E_S` and `nu`.** An empty intersection obstructs only that
   fixed selector. Dynamic reselection or an event-level certificate can
   escape the local conclusion and remains open.
10. **Many-to-one windows are not merged stones.** S36 lies outside S22 and
    S8.1 because it has no point map. It does not authorize a noninjective
    total embedding inside those theorems' scopes.
11. **S38 is transfer, not maintenance.** It proves P5R for complete traces
    that maintain the certificate. S37 supplies one exact local maintenance
    test, not a universal handler.
12. **S39.1 is a scoped obstruction.** It applies at common-phase Shat
    decision nodes consistent with an alleged winning `sigma`. Phase-lagged,
    multi-step, dynamically reselected, and earlier-terminal representations
    are not collapsed to `C_shield`.

## 42. Status ledger and objective verdicts

### 42.1 New result ledger

| Claim | Status | Exact scope and agenda advance |
|---|---|---|
| S32 fixed-total two-escape obstruction | **PROVEN** | Zero-lag total-exact owner-faithful translation/D6 isometric subbranch of (A), at most two S-reactive episodes over the whole continuation; Agenda 1/2/7/10 |
| S32 own-history reachability | **PROVEN** | Three cuts recomputed around two genuine `sigma` pairs; Agenda 10 |
| Per-S-pair budget `K=2` | **OPEN** | Two episodes can finish the tested pair; no third S choice occurs before F acts |
| S33 phase-exact rolling admission | **PROVEN** | First lag iff post-deficit at least two; every nonwinning second lag; exact `tau_E<=2` service; Agenda 3/4 |
| S34 rolling guarded-lag invariant | **PROVEN at scope** | Every finite `A_FS2` continuation; preventive P5R and deterministic real-board service; Agenda 2/3/4/8/9 |
| S34 P3 composition | **OPEN** | Requires a genuine carrier for arbitrary `sigma` and same-step P5 certificates; Agenda 1/6/7 |
| S35 active reconciliation/service | **PROVEN** | Exact finite legal trace, `tau_E=1`, legal strategy not claimed winning; Agenda 3/4, not 10 |
| Definition 39.1 window-deficit representation | **Definition** | Physical many-window-to-one P5R certificate; no cell map; Agenda 3/8 |
| S36 nonisometric witness | **PROVEN** | Reachable physical checkpoint; no injective point map induces its two assignments; Agenda 3 |
| S37 maintenance algebra | **PROVEN** | Fixed `E_S,nu`, one associated S/`Shat` append; Agenda 2/3 |
| S38 deficit-certificate P5R | **PROVEN at scope** | Complete traces maintaining post-append physical certificates; Agenda 3 |
| S39 winning-strategy barrier | **PROVEN** | Genuine live Shat node consistent with alleged-winning `sigma`; Agenda 3/8/10 |
| S39.1 natural window-recode obstruction | **PROVEN** | Definition 39.1 implies `delta_R>m`; broader event certificates remain open |
| Global P0--P6 plus P5R coupling | **OPEN** | No one mechanism satisfies all ten review items for every alleged-winning `sigma` |
| `NL_F` | **OPEN** | D2 remains available; neither determinacy alternative is selected |

There are no **SKETCH** or **CONJECTURE** claims in this round.

### 42.2 Verdict by requested objective

1. **Sharp question: PROVEN-AT-SCOPE, globally PARTIAL.** Negative side: S32
   refutes `C_A^{K=2,total}`, the total-exact isometric subbranch of (A), with
   fixed lifetime `K=2`. Positive side: S34 constructs a rolling branch-(B)
   mechanism and proves its invariant against the exactly
   named finite `A_FS2` continuation class. Arbitrary P3 composition and
   per-pair `K=2` remain open.
2. **Window-certificate representation: PROVEN-AT-SCOPE.** Definition 39.1
   requires no total point map, and S36 proves a member not point-induced by
   any injection. S38 transfers P5R, while S39.1 is the first obstruction to
   using this natural family to carry shield-unsafe debt.
3. **Ledger maintenance: PROVEN.** Sections 40--42 update every item in the
   authoritative ten-item agenda and retain all inherited regressions.

## 43. Provenance and next question

**Input state at start.** Branch `hunt/gap-raw`, HEAD `67f996d1`. This
authoring pass creates no commit. During authoring, an external concurrent
update advanced the branch and `origin/hunt/gap-raw` to `110042e5`; a final
read-only name diff showed that update touched only
`.codex-gr/prompt-tempo-round6.txt` and `GAP_RAW_PROOF_ROUND6.md`. This pass did
not create, amend, reset, or otherwise alter that commit. The required
stealing corpus and cited engine sources were unchanged by it.

The only intended deliverable written in this pass is
`STRATEGY_STEALING_ROUND5.md`; no production source or `GAP_RAW_*` file was
edited by this pass.

**Required corpus read first, in order and in full.**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including its folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md`, including folded errata section 24, then
   `STRATEGY_STEALING_REVIEW_ROUND3.md`; and
4. `STRATEGY_STEALING_ROUND4.md`, including binding folded errata section 35
   and the corrected section-34.1 quantifiers, then
   `STRATEGY_STEALING_REVIEW_ROUND4.md`.

**Rule sources checked.** The cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs` were
read directly. In particular this pass verified `state.rs:149-160` for the
initial owner/phase and `state.rs:283-288` for the analysis/MCTS context of
apply/undo, in addition to the forward transition and terminal ranges cited
in section 36.2.

**Machine work.** None. No Cargo command, Lean build, harness, executable
search, or proof search was run. No commit was created.

**Single sharpest next question [OPEN].** For every alleged-winning `sigma`
and every rolling `A_FS2` prefix, can one genuine causal P3 carrier realize
the canonical service pair `K_E` while also surviving S18, transferring every
S20-type `Fhat` terminal placement, and avoiding S12--or can failure of such a
carrier be forced on its own `sigma`-consistent history?

## 44. Errata and clarifications folded from the round-5 hostile review

`STRATEGY_STEALING_REVIEW_ROUND5.md` (ultra, reviewed artifact `d0af2ef4`)
returned **SOUND-WITH-MINOR-ERRATA**: no REFUTED or MAJOR finding; all three
round objectives CONFIRMED (S32 at exact lifetime/total scope; the S34
finite conditional `A_FS2` invariant; the S36--S39.1 window-certificate
module). The following repairs are folded inline above; this section records
them.

1. **(Finding 1, NOTE)** Section 36.2's source list now includes
   `tactics.rs:21-75` — the three Q/R/QR axis-vector window families used by
   the section-38/39 window constructions.
2. **(Finding 4, MINOR)** Grammar nonemptiness of `G_A^2` is now witnessed by
   a deterministic candidate `C_0` (fixed-enumeration tie-breaks, first
   failed promise reported off-path), not by the round-4 trace alone: a
   legal partial execution is not itself an algorithm. S32 and the emptiness
   of `C_A^{K=2,total}` are unaffected.
3. **(Finding 6, MINOR)** The S34 induction now exposes the transient
   two-debt microstate between real `S@y` and shadow `Shat@T(e)`: for every
   old E-live window, `delta'-m' = (delta-m)+1-1_{y in W} > 0`, so no
   old-debt window can become terminal in the transient physical real state.
4. **(Finding 7, MINOR)** Definition 38.1's terminal sentence and `A_FS2`
   clause 1 now classify a *filler-created* physical `Shat` win: any
   branch-(B) `Shat` append, reconciliation or filler, ends the physical
   module when it wins; the win contradicts `sigma` only under an outer
   P0/P3 carrier.
5. **(Finding 15, MINOR)** S35 is a "P3-shaped, phase/legality-paired
   physical trace," not "P3-compatible" for arbitrary alleged-winning
   `sigma`. Its agenda credit moved to the geometry side of items 3--4 and
   physical persistence in item 8; item 1's only round-5 advance is S32's
   negative crossing. Arbitrary positive P3 composition remains OPEN.
6. **(Finding 16, MINOR)** The regression ledger now names the inherited
   round-2 regressions: S13 (fixed-isometry one-stone FIFO frontier failure)
   in agenda items 2/6, and S14 (literal one-S-stone-lag terminal-count
   failure) in item 3 and the P5R cross-ledger row. Neither is silently
   solved: the rolling queue differs from S13's schedule by persistent
   genuine fillers/proxies and physical reconciliation, and S34 avoids S14
   only on shield-admitted traces.

**Review confirmations of record.** Per-pair `K=2` remains OPEN (Finding 5);
the review's checkpoint table (Finding 2) and terminal-fork exhaustiveness
(Finding 3) independently recompute S32; S36's seventeen-window census,
deficit-three cover, and injectivity contradiction recompute exactly
(Finding 12); S38 implements S26's same-step physical terminal certificate
and S39.1 scope-limits rather than contradicts S34 (Finding 14). The
review's twelve-item unresolved-obstacle list supersedes and refines
section 43's obstacle statement as the authoritative open state.
