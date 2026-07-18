# Strategy stealing in engine Hexo, round 7: co-terminal alignment

**Worktree:** `hunt/gap-raw` at requested input HEAD `09e27a93`  
**Date:** 2026-07-18  
**Alignment verdict:** **PARTIAL** -- **CONDITIONAL POSITIVE-AT-STRICT-
SUBCLASS** for the augmented exact selector of section 55; the universal
canonical and alleged-winning-`sigma` alignment questions remain **OPEN**.  
**Global target:** `NL_F` remains **OPEN**.

Round 6 proved that temporal event pairing solves the nonterminal F-role P3
problem on finite `A_FS2^ET(sigma)` traces. This round separates the last
semantic premise from the trace class that already assumes it. The resulting
physical invariant is the **Causal One-Debt F-Window Alignment Cover**, written
`F-CAD_2^st`. It assigns each near-terminal shadow-`Fhat` window an actual
unblocked real-F window and tracks the exact difference of their physical
deficits. The update law is one line of append-only arithmetic. State
maintenance alone does not choose the required terminal service cell. For the
canonical rule that residual selector duty is isolated as `F-LOCK`; for the
augmented exact rule, readiness lets the handler choose the assigned real
window's unique hole after the reached shadow prescription is known.

This is a genuine conditional positive result for the augmented handler, but
not a universal one. The inherited
canonical service does not maintain `F-CAD_2^st` on every first-safe,
two-serviceable history. More strongly, after S15 the shadow F role always has
one more stone than real F. If the shadow wins with its sixth physical stone,
the paired real event leaves F with only five stones, so no choice, ordering,
or augmentation of the two existing service placements can be co-terminal.
S42 is an exact legal-strategy instance. No theorem forces that fast terminal
age from a globally winning `sigma`, so this does not close the alleged-
winning-only negative route.

The membership analysis supplies one new covered ejection class. When the
strict rolling queue rejects an old certificate because `T(e)` is already
occupied by the correct shadow role, that physical stone really certifies
`e`. If the current image is fresh and legal, exact copying gives a branch-(A)
continuation; otherwise a guarded filler rotation gives a branch-(B)
continuation. Wrong-role occupancy and fresh-but-unsupported certificates
remain open. Finally, an explicit alternating-translation cylinder shows that
two intra-pair rebindings can be executed legally through both displayed
coordinate episodes. The same cylinder also admits fixed `T_0`, so it is a surviving
construction at finite scope, not a separation from S43.1 and not a universal
per-pair-`K=2` carrier.

All results below are hand proofs. No Cargo command, Lean build, harness,
executable search, or proof-search program was run. This pass creates no
commit.

## 54. Statement boundary and binding inherited state

### 54.1 Roles, target, and status discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real second
player. In the role-swapped shadow, `Shat` represents real S and is the
opener, while `Fhat` represents real F and follows a fixed pure second-player
strategy `sigma`. The target is unchanged:

`NL_F : exists pure sigma_F, for every pure sigma_S, S never wins`.

As in rounds 1--6, `S never wins` permits either a finite F win or an infinite
nonterminal history with neither owner completing six. Round-2 Theorem D2 is
inherited **[PROVEN from the CITED Gale--Stewart open-determinacy theorem]**
for the declared unbounded-board macro-game:

`NL_F <=> S has no winning strategy`.                         (D2)

This round does not refute every alleged-winning `sigma`, so D2 does not yield
`NL_F`. Every named or load-bearing claim is marked **PROVEN**, **SKETCH**,
**CONJECTURE**, **OPEN**, or **CITED**. Definitions are stipulative.
`PARTIAL`, `CONDITIONAL POSITIVE-AT-STRICT-SUBCLASS`, and
`NEGATIVE-AT-SCOPE` are objective dispositions, not proof statuses. There are
no machine-verified claims.

### 54.2 Production rules used [PROVEN]

On axial coordinates the engine distance is

`d((q,r),(q',r'))=max(|q-q'|,|r-r'|,|(q-q')+(r-r')|)`.

The only opening is `F@(0,0)`. Along nonterminal forward play the cadence is

`F ; S,S ; F,F ; S,S ; F,F ; ...`.

A normal coordinate is legal exactly when it is physically empty and belongs
to the color-blind radius-eight legal store. A nonwinning first placement is
inserted before the second placement is validated. A Q-, R-, or QR-axis
length-six window is checked after every single append and before phase
advancement. A winning first placement suppresses the second; a terminal
state has no legal continuation.

These facts are implemented in
`packages/hexo_engine/rust/src/coord.rs:1-4,9-20,76-95`,
`legal.rs:17-18,123-145`, `rules.rs:11-44`, `board.rs:83-105`,
`state.rs:149-160,203-252,289-357`, and
`tactics.rs:13-17,21-75,205-208,451-485`. Forward placements insert physical
occupancy and history. The apply/undo API is an analysis/MCTS facility, not a
legal-history operation. Every old proxy, filler, service stone, and queued
stone therefore persists in occupancy, support, blocking, and terminal
windows.

The executable carrier is `i16`. General theorems retain the inherited `Z^2`
idealization. Every displayed finite coordinate cylinder below is separately
inside a small safe region.

### 54.3 Exact round-6 inheritance

The following statements and caveats are binding.

- S40 proves genuine, causal, sequential P3 event pairing on the terminal
  closure `A_FS2^ET(sigma)`. That class already requires every terminal
  shadow-`Fhat` append to have a same-step real-F terminal append.
- Section 53's terminal-closure definition supersedes round-5 clauses 5--6 on
  the final paired F microstep: both associated physical appends occur inside
  that event before the trace closes.
- S41 is one complete aligned trace passing S18 and a proxy-assisted terminal
  stress case. Its `sigma_star` is legal but not proved globally winning.
- S42 refutes the named terminal-blind selector on its own legal
  `sigma_dagger` history. It is not a negative theorem for an alleged-winning-
  only domain.
- S43 proves that two episode charges exactly saturate one S pair and then
  reset. S43.1 excludes only the subclass whose translation/D6 isometry stays
  fixed within that pair.
- S44/S44.1 prevent a common-phase deficit certificate from discounting
  missed physical service at fixed debt. S45 gives exact `tau_E=5` for S30.
- The round-6 review's twelve-item obstacle list is the authoritative open
  state. In particular, intra-pair changing isometries, total nonisometric
  point recodings, common-only real wins, and simultaneous legality/terminal
  maintenance remain open.

It would be circular to announce an induction *on `A_FS2^ET`* as the missing
alignment proof: event-terminal alignment is part of that class's definition.
Section 55 therefore removes the implication first, defines the physical state
invariant, proves its exact update on admitted raw traces, and only then embeds
the certified subclass back into `A_FS2^ET` **[PROVEN as a definition-level
necessity]**.

### 54.4 Branch quantifiers, unchanged

For every alleged-winning `sigma`, every strategy-generated genuine live
prefix, and every observed legal real-S single placement, a global candidate
may select exactly one local response:

- **(A)** a zero-lag repair completed before the next engine placement;
- **(B)** an explicit lag/queue satisfying its phase-sensitive P5R guard; or
- **(C)** an actual same-step physical shadow-`Shat` terminal certificate.

These are alternatives only for that S placement. In **every** branch the
candidate must also provide recurring P3 transfer, physical service or
reconciliation, P5 and P5R terminal fidelity, causal selection, one genuine
append-only `sigma`-consistent history, and accounting for all persistent
stones. S12, S13, S14, S18, S20, S25, S30, and S31 remain mandatory at their
inherited scopes. A negative gadget counts against a strategy-specific
candidate only when selected on that candidate's own legal coupled history.

## 55. The alignment question: a physical F-window deficit invariant

### 55.1 Remove the conclusion from the carrier class

**Definition 55.1 (`A_FS2^EV(sigma)`, raw event closure).** Start with live
rolling `A_FS2` segments and generate every F-role step by the physical paired
events of round-6 Definition 46.1. Retain the section-53 atomic final-event
semantics, but omit implication (46.4). Take the terminal closure with these
four physical outcomes:

1. two nonterminal appends continue subject to the next handler clauses;
2. a real-F terminal append with a nonterminal shadow append is a sound real
   stop;
3. two terminal F-role appends are an aligned stop; and
4. a shadow-`Fhat` terminal append with a nonterminal real append is a closed
   **misaligned terminal test extension**.

A `Shat` win on a certificate or filler is a different physical stop. A
failed old-debt certificate is not promoted to a successful raw continuation;
it is a membership exit treated in section 57. Thus

`A_FS2^ET(sigma) subseteq A_FS2^EV(sigma)`,

and equality is precisely the missing F-terminal question, not a premise of
the raw class.

### 55.2 Alignment debt

Let `W_6` be the physical engine six-window family. At a live pre-F-event
state define

`U_R^F={W in W_6 : W intersect X_S = empty}`,               (55.1)

`U_H^F={V in W_6 : V intersect X_Shat = empty}`.             (55.2)

For these opponent-unblocked windows put

`delta_R^F(W)=6-|W intersect X_F|`,                          (55.3)

`delta_H^F(V)=6-|V intersect X_Fhat|`,                       (55.4)

and for a selected pair `(V,W)` define its **alignment debt**

`a(V,W)=delta_R^F(W)-delta_H^F(V)`.                          (55.5)

At a common-live state every displayed deficit is positive. The family of
shadow windows with `delta_H^F<=2` is finite: each contains at least four of
the finitely many physical `Fhat` stones, and each stone belongs to eighteen
engine windows.

**Lemma S46 (exact paired-event debt update) [PROVEN].** Let a legal paired
F-role event append `Fhat@z / F@k`. For any pre-event
`V in U_H^F` and `W in U_R^F`, their opponent-unblocked status persists through
this F-role event and

`a'(V,W)=a(V,W)-1_{k in W}+1_{z in V}`.                     (55.6)

*Proof.* The real append reduces `delta_R^F(W)` by one exactly when `k` is in
`W`; the shadow append reduces `delta_H^F(V)` by one exactly when `z` is in
`V`. Neither F-role append adds an opponent stone, so neither window becomes
opponent-blocked. Subtract the two updated deficits. This is physical
append-only occupancy arithmetic; no coordinate map is used. QED.

Debt one is the exact catch-up state. If `delta_H^F=2` and `a=1`, the real
window may have deficit three. An event with `k in W` but `z notin V` repays
the debt to zero. An event hitting both windows preserves the debt. Once the
shadow window has deficit one, debt one is too late: the paired real window
would still have deficit two.

### 55.3 The state invariant, the residual lock, and augmented exact service

**Definition 55.2 (Causal One-Debt F-Window Alignment Cover,
`F-CAD_2^st`).** Fix a portfolio policy `Pi` before the coupled trace. At every
live pre-query F-role microstep, `Pi` receives only the reached physical prefix
and returns a portfolio assigning to each

`V in U_H^F` with `delta_H^F(V)<=2`

one real window `pi(V) in U_R^F`. The assignment may be many-to-one and must
satisfy:

1. **one-debt cover:** `a(V,pi(V))<=1`; and
2. **terminal readiness:** if `delta_H^F(V)=1`, then
   `a(V,pi(V))<=0`.

The returned portfolio `pi_i` is fixed before `sigma` is queried for `z_i`; it
may not be selected retroactively after seeing that prescription. After a nonwinning first paired
event, (55.6) updates every retained pair, the physical near-window families
are recomputed, and a new `pi_2` must be selected from that reached prefix
before querying `z_2`. After an intervening S/`Shat` turn the next portfolio
is reselected from the full observed physical prefix, because opponent stones
may have blocked old windows. This is the maintained **state** invariant. It
does not itself say which real service coordinate is selected.

The S15 base has an empty portfolio because `Fhat` owns only two stones. On a
nonterminal paired event Lemma S46 gives the exact inductive update for every
retained assignment; admitting a newly created window, or finding a new
assignment after an S/`Shat` turn, is an explicit feasibility obligation. A
raw trace is in `A_FS2^{CAD-st}(sigma;Pi)` precisely when one fixed causal
`Pi` returns an admissible pre-query portfolio at every reached F microstep in
addition to the inherited live handler conditions. The notation without
`;Pi` takes the union over policies fixed in this way. Thus the invariant is
inductive on that strict trace class, not asserted on all of `A_FS2^EV`.

**Definition 55.3 (canonical event lock, `F-LOCK(pi_i)`).** Let the inherited
canonical service coordinate `k_i` already be fixed as in round-6 Definition
46.1. After the reached prescription `z_i` is known, if it completes one or
more covered shadow windows, `F-LOCK(pi_i)` requires

`exists completed V_* : k_i in pi_i(V_*)`.                 (55.7)

This is deliberately separate. It is the remaining selector obligation in
window language, not a consequence of the debt update. Write
`A_FS2^{CAD+LOCK}(sigma)` for canonical raw traces maintaining
`F-CAD_2^st` under one such `Pi` and passing (55.7) at every terminal shadow
event. This is a sufficient class, not a necessary characterization: a real
service cell might win in an unassigned window.

For comparison define a separate sequential augmented handler
`svc_{E+CAD}`; its raw and aligned closures are denoted
`A_FS2^{EV,+}(sigma)` and `A_FS2^{ET,+}(sigma)`. At a live pre-query
microstep it first evaluates its fixed `Pi` to obtain `pi_i`, then queries the
actual legal `z_i`. For this exact handler, fix well-orders of engine windows,
finite portfolios, and legal coordinates, and let `Pi` return the least
admissible portfolio. The candidate set is finite: every assigned `W` has
`delta_R^F(W)<=delta_H^F(V)+1<=3`, hence contains a physical real-F stone;
each of finitely many such stones lies in only eighteen engine windows. Thus
the least selection is a genuine pure rule, not a causal relation.

- If `z_i` is terminal, take the least completed `V_*` in a fixed enumeration,
  put `W=pi_i(V_*)`, and choose the unique real hole of `W`. No later urgent
  transversal is owed after this sound terminal stop.
- If a first `z_i` is nonterminal, choose the least legal real `k_i` such that,
  unless `k_i` itself wins, the residual inherited family
  `R_i(k_i)={W in U_E : k_i notin W}` has a currently empty cell that will be
  legal after `k_i` and hits every member, and the post-event prefix admits a
  least `F-CAD_2^st` portfolio before `z_2` is queried.
- If a second `z_i` is nonterminal, choose the least legal real `k_i` which
  completes the unresolved E-urgent service; if that family is empty, choose
  the least legal real filler. A real win is a sound stop. Continued admission
  at the next F turn is tested only after the intervening S/`Shat` handler.

If a required nonterminal admissible set is empty, the augmented branch exits
its strict class; no availability theorem is being smuggled into the
definition. The second prescription is queried only after both first appends
are verified nonterminal. These choices use a reached prescription but no
future S action, so they are causal and outside S12's preannouncement premise
**[PROVEN from the event order]**. Formally, `A_FS2^{EV,+}` is Definition 55.1
with only Definition 46.1's canonical `svc_E` replaced by this sequential
selector; all live `A_FS2` clauses, the atomic paired-final-event semantics,
and the same four raw outcomes are retained. `A_FS2^{ET,+}` removes only raw
outcome 4 by imposing co-terminal alignment. Because this is a different
service rule, its traces are not relabeled as inherited canonical traces.

**Theorem S47 (conditional alignment transfer and augmented exact alignment)
[PROVEN].** For every pure strategy `sigma`:

1. `A_FS2^{CAD+LOCK}(sigma) subseteq A_FS2^ET(sigma)`; and
2. every trace of `A_FS2^{EV,+}(sigma)` on which `svc_{E+CAD}` remains
   admitted to a shadow terminal event belongs to `A_FS2^{ET,+}(sigma)`.

*Proof.* Fix a reached terminal prescription and choose the event-lock witness
`V_*` in part 1, or the least completed `V_*` selected by the augmented rule in
part 2. Immediately before the append, `V_*` is `Shat`-unblocked, has deficit
one, and the prescription is its unique physical hole. Readiness gives

`delta_R^F(pi_i(V_*))<=delta_H^F(V_*)=1`.

The real board is live, so this real deficit is also at least one and is
therefore exactly one. In part 1, canonical `k_i` lies in that window by
`F-LOCK`; because it is a legal empty coordinate, it is the unique hole. In
part 2 the augmented handler chooses that hole directly. The cell is engine-
legal: it is empty and lies within distance at most five of one of the five
real-F stones already in the length-six window. The real append therefore
fills all six cells on the same coupled microstep.

Induct over reached F events. The base portfolio is vacuous. After two
nonwinning first appends Lemma S46 and strict admission provide the next
pre-query portfolio before either second prescription is requested. A terminal
first placement suppresses the second placement on that terminal board, and
the coupled event closes by its stated stop rule. Two nonwinning seconds pass
both histories to the S role. A real-only win is an earlier sound stop. Hence
every reached shadow terminal event in either stated class is co-terminal.
QED.

Part 1 is a conditional certificate normal form: `F-LOCK` isolates, rather
than solves, the canonical selector bridge. Part 2 is a genuine augmented
terminal rule derived from readiness, but nonterminal service compatibility
and recurring portfolio admission remain premises. S47 therefore delegates
portfolio existence after adversarial S/`Shat` placements, old-certificate
freshness, and all coverage outside the strict class. S41 below proves a
complete canonical `CAD+LOCK` terminal trace; nonemptiness of a complete
augmented terminal trace under the exact least-choice rule is not separately
proved **[OPEN]**.

### 55.4 Positive and negative audits of the invariant

**Lemma S48 (S41 debt repayment and S42 red line) [PROVEN].** The S41 terminal
trace admits `F-CAD_2^st` and its canonical events satisfy `F-LOCK`, while the
S42 trace violates terminal readiness before its final prescription.

*Proof: S41.* Use

`V={(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)}`

and

`W={(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)}`.

After S41's seed service, `V` has shadow deficit two and `W` has real deficit
three, so `a(V,W)=1`. Service 1 is off both windows and preserves that debt.
At service 2 the event `z=(-8,0), k=(3,0)` has `z notin V` and `k in W`, so
(55.6) repays the debt to zero. The following event
`z=(5,0), k=(4,0)` hits both windows, preserves zero, and leaves both deficits
one. The intervening rolling pair blocks neither window. Before `z=(5,0)`,
`V` is the sole unblocked deficit-at-most-two q-window. After that event the
shifted window `{(2,0),...,(7,0)}` also has deficit two and may use the same
real `W`, whose deficit is then one; many-to-one assignment is allowed.
Off-line `Fhat` stones form runs of length at most two, and q-windows
containing `(0,0)` are `Shat`-blocked, completing the near-window census. The
final event `z=(6,0), k=(5,0)` fills `V/W` and satisfies `F-LOCK`. These
prefix-indexed assignments define one causal `Pi` on the displayed finite
trace; no assignment uses the as-yet unqueried prescription.

*Proof: S42.* Immediately before the second post-S15 service pair, its shadow
q-window has deficit two while the displayed real q-window has deficit three.
The first event `z=(5,0), k=(3,0)` advances both. The shadow window now has
deficit one, while real F has only four stones in total; hence every real
six-window has deficit at least two. Terminal readiness fails for every
possible portfolio assignment before `z=(6,0)`. That second prescription wins
only in the shadow, exactly as S42 proved. QED.

Consequently ordinary first-safety, certificate freshness, and
`tau_E<=2` do not imply `F-CAD_2^st`. S42 stays inside all three tests with
`tau_E=0`. The existing canonical service rule therefore does **not**
maintain the state invariant on every `A_FS2` trace **[PROVEN]**. S41 proves
that the canonical `CAD+LOCK` terminal class is nevertheless physically
nonempty and passes the S18 inverse-legality stress and the proxy-assisted S20
phenomenon (not a literal second-placement S20 witness).

## 56. The adaptive negative route

### 56.1 A selector-independent terminal-age barrier

**Theorem S49 (sixth-`Fhat`-stone alignment barrier) [PROVEN].** Fix any pure
`sigma`, any S15 synchronization for it, and any genuine one-for-one F-role
event continuation with no unmatched real-F placement. If a shadow-`Fhat`
terminal event is `Fhat`'s sixth physical stone, its paired real append cannot
be a real-F terminal certificate. This remains true for every choice,
ordering, or augmentation of the two existing real service placements.

*Proof.* At S15 synchronization the physical counts are

`|X_Fhat|=2` and `|X_F|=1`.

Each later paired F event adds exactly one physical stone to each role, so

`|X_Fhat|=|X_F|+1`                                           (56.1)

at every reached event. When the shadow appends its sixth `Fhat` stone, the
paired real append leaves F with exactly five stones. Five physical stones
cannot fill a six-cell window. Before that event real F had at most four, so
there was no already-installed real-F six either.

In the synchronized cadence the event is necessarily the fourth post-S15
prescription, the second placement of the second post-S15 F turn. Its first
placement leaves shadow/real F-role totals five/four and is necessarily
nonwinning. Reordering the two service coordinates cannot change either
count, and a third real placement is forbidden by the engine cadence. QED.

S49 applies, in particular, to every alleged-winning `sigma` **if its own
admissible carrier history has this fast terminal age**. S42 supplies a fully
legal first-safe, certificate-valid, `tau_E=0` instance for `sigma_dagger`.
Because `sigma_dagger` is not proved winning against every `Shat`
counterplay, S49 does not establish that the alleged-winning-only domain
contains such a trace.

**Corollary S49.1 (no universal canonical invariant) [PROVEN].** For the S42
canonical selector there is no prefix property which (i) holds at every S15
base, (ii) is preserved through every full live canonical `A_FS2` handler
extension for every legal pure `sigma`--real-S appends, filler/certificate
reconciliation, and nonterminal paired F events included--and (iii) implies
co-terminal alignment at the first shadow-`Fhat` terminal event.

*Proof.* Follow S42. Its live prefixes meet (i) and every hypothesized
preservation instance in (ii), but its sixth-`Fhat`-stone extension violates
(iii) by S49. QED.

This rules out an unconditional invariant for the existing canonical rule;
it does not rule out a property preserved only on alleged-winning-strategy
histories or the strict `F-CAD_2^st` admission class.

### 56.2 One adaptive cycle for every fixed strategy

The S42 cylinder uses one specially chosen legal strategy. The next theorem
reaches the same earliest terminal test on the actual behavior of an
arbitrary fixed strategy, but it does not force that behavior to be terminal.

**Theorem S50 (adaptive earliest-cycle dichotomy) [PROVEN].** On `Z^2`, for
every pure strategy `sigma` and every S15 synchronization for it, there is a
legal causal real-S continuation which:

1. passes the next `FirstStone` safety test;
2. supplies a fresh legal old-debt certificate on `SecondStone`;
3. ends its pair with `tau_E<=2`;
4. protects the next old-debt coordinate from both reached prescriptions of
   `sigma`; and
5. reaches `sigma`'s fourth post-S15 prescription on a genuine raw event
   history.

At that prescription exactly one of the following occurs:

- it wins for `Fhat`, in which case S49 forces terminal misalignment on
  `sigma`'s own history; or
- it is nonwinning, in which case both boards survive the earliest stress
  cycle and the protected old certificate remains fresh and shadow-legal.

*Proof.* The first post-S15 `sigma` pair and its paired real service are
necessarily nonwinning: they raise the F roles only from shadow/real counts
two/one to four/three. Thus the carrier reaches a common S `FirstStone` node.
Let `a` be either represented real-S stone from the S15 pair. Choose

`y_1 in B_8(a) \ (O_R union T^{-1}(O_H))`.

The radius-eight ball has 217 cells, while at this checkpoint the two physical
occupancies have sizes five and seven. Hence at least `217-5-7=205` choices
remain. The selected `y_1` is real-empty and supported by `a`; `T(y_1)` is
shadow-empty and supported by physical `Shat@T(a)`. Real S then has only three
stones, so `y_1` is nonwinning and every F-unblocked window through it has
deficit at least three. It is first-safe.

With empty debt the rolling handler appends a fresh legal filler
`f!=T(y_1)`. Such a filler exists: even the ball `B_8(T(a))` contains at least
`217-7-1` eligible cells. It gives `Shat` only its fourth stone and is
nonwinning. The handler's next shadow placement is now fixed as
`Shat@T(y_1)`; it is fresh, legal, and gives `Shat` only five stones.

Because `sigma` is a fixed pure function, the shadow history after that fixed
append determines its next first prescription `z_1` and, since `z_1` raises
`Fhat` only to five stones, its reached second prescription `z_2`. Before the
real second coordinate is chosen, select

`y_2 in B_8(y_1)`

outside the current real occupancy and outside the inverse images of the
current shadow occupancy, `z_1`, and `z_2`. Fewer than eighteen cells are
excluded from a 217-cell ball, so such a coordinate exists. It is legal at
real `SecondStone`, differs from occupied `y_1`, and gives real S only four
stones, hence is nonwinning. Execute the already fixed physical certificate
`Shat@T(y_1)` and leave `E_S={y_2}`.

Any urgent window for this final debt must contain at least four real-S
stones. There are exactly four total, so all four lie in that window and are
collinear on one axis. Four distinct indices in a six-cell interval have span
three, four, or five. For span five there is one urgent interval; for span
four there are at most two; for span three the four cells are consecutive and
the three possible hole pairs have a two-cell transversal, exactly the
one-axis pattern used in S45. Thus the whole urgent family has
`tau_E<=2`. The canonical service pair is legal.

The next shadow prescriptions are exactly the precomputed `z_1,z_2`; the
choice of `y_2` did not alter the shadow history. Avoiding their inverse images
makes `T(y_2)` remain fresh. It stays shadow-legal through physical support
from `T(y_1)` at distance at most eight. The first prescription gives
`Fhat` only five stones and cannot win. At the second, shadow/real F-role
counts are six/five. If it wins, apply S49; otherwise both boards and the next
certificate remain live. QED.

The counterplay's look-ahead is causal in the game-theoretic sense used by
S12: `sigma` is fixed and pure, and the intervening shadow move is already
determined. The actual carrier still queries `z_1` and `z_2` only at their
engine phases. No private value is learned from a future opponent choice.
This theorem concerns the unbounded idealization; it also applies to any
finite instance whose displayed selections remain in the safe carrier.

S50 is genuine membership pressure on an arbitrary strategy's own history.
It is not yet a negative theorem for alleged-winning `sigma`: winningness
guarantees a finite terminal horizon, but not that the horizon is four
post-S15 prescriptions.

**Lemma S50.1 (the earliest threat coercion has a two-cell negative control)
[PROVEN].** At the common shadow `FirstStone` checkpoint immediately before
the second post-S15 F pair--before its third prescription--the shadow opener
has exactly five stones. Its family of immediate winning cells has size at
most two and has a two-cell transversal. If there are two cells, the five
stones are consecutive on one axis and the two cells are the outside
endpoints, at hex distance six. `Fhat` can occupy both during that pair, and
those two blockers cannot themselves complete an `Fhat` six. If there is only
one threat, `Fhat` can block it and use a nonwinning legal padding cell.

*Proof.* A one-hole `Shat` window uses five `Shat` stones, hence at this count
uses all of them. Two distinct such windows must lie on the same axis and
share those five cells; this happens only for five consecutive cells, giving
the two shifted length-six intervals and their outside endpoints. Otherwise
there is at most one hole. Both endpoints are legal through the five-stone
line and are distance six apart. Before the pair `Fhat` has four stones; if
it uses both blockers, any six made by its sixth total stone would have to
contain both new stones, but two cells of one engine six-window are at
distance at most five. Therefore the blocking pair is nonwinning.

In the one-threat case, after the blocker `Fhat` has five stones. By the same
five-stone window argument it has at most two immediate winning cells. A
radius-eight ball around any `Fhat` stone has 217 cells, while the shadow
occupancy and those at most two cells exclude at most twelve; choose a
different supported empty cell. It is legal and nonwinning, supplying the
padding claim. QED.

Thus a simple attempt to compel the fast S49 terminal by giving `Shat` more
immediate threats than F can block cannot work at the earliest count. When the
immediate-threat family is nonempty: if `sigma` wins instead, S49 applies; if
it blocks, the current F pair remains nonterminal; if it neither wins nor
blocks, the next `Shat` win refutes the alleged-winning premise.

### 56.3 Finite-horizon reduction and the exact circularity boundary

**Theorem S51 (finite-horizon stop trichotomy) [PROVEN].** Fix an
alleged-winning `sigma` and a genuine live coupled checkpoint `h` whose shadow
component is legal, reachable, nonterminal, and `sigma`-consistent. Let
`N_sigma(h)` be S24's finite upper bound on the length of a nonterminal shadow
continuation compatible with `sigma`. For every causal outer continuation
attempt, within that shadow horizon at least one of the following occurs:

1. raw preterminal `A_FS2^EV` membership or another mandatory branch
   obligation *other than the tested `Fhat`-to-real-F terminal-fidelity duty
   P5, equivalently implication (46.4),* fails;
2. real F reaches a sound physical stop no later than `sigma`, including an
   aligned terminal event; or
3. `sigma` reaches a shadow-`Fhat` terminal event while real F remains
   nonterminal.

Accordingly, if one constructs a continuation avoiding items 1 and 2 through
the horizon, item 3 is forced on `sigma`'s own legal history.

*Proof.* Under avoidance of item 1, the outer rule supplies one genuine legal
append-only shadow continuation compatible with `sigma`. A shadow-`Shat` win
is impossible under the alleged-winning premise. If item 2 never occurs and
`Fhat` also stays nonterminal through `N_sigma(h)`, the resulting compatible
nonterminal prefix contradicts the definition of S24's bound. Hence `Fhat`
terminates; without item 2 the paired real board is nonterminal, which is item
3. QED.

S51 isolates the missing constructive burden: keep the trace covered and the
real canonical service nonwinning until the strategy-dependent horizon. It
does not supply that real-S controller.

**Corollary S51.1 (branch-(C) coercion is already a refutation) [PROVEN].** At
a genuine common-live `Shat` decision node consistent with alleged-winning
`sigma`, no legal `Shat` first or second placement may complete a physical
six. Equivalently, every `Fhat`-unblocked shadow window has deficit greater
than the number `m` of `Shat` placements remaining in the turn.

*Proof.* This is S39 at its inherited scope. A legal same-turn completion
extends by least-legal off-path moves to a total counterstrategy defeating
`sigma`. QED.

Branch (C) is therefore a valid successful contradiction stop, but it cannot
be used as a continuing device while still calling `sigma` alleged-winning.
The broader negative route is **not** logically circular: feeding `sigma` a
legal nonwinning `Shat` counterplay is exactly what a winning strategy must
handle. It becomes a refutation only if `Shat` wins or play stays nonterminal
forever. What remains **OPEN** is the physical membership-and-real-stop
avoidance construction required by S51.

## 57. Membership pressure outside strict `A_FS2^ET`

### 57.1 Exact rolling exit partition

At a post-placement real state let, for a legal `S@y`,

`d_y=min{6-|W intersect X_S| : y in W, W intersect X_F=empty}`, (57.1)

with value infinity if the set is empty. At `FirstStone`, the branch-B
admission test is `d_y>=2`; at `SecondStone`, nonterminality is `d_y>=1`.

**Theorem S52 (strict membership partition) [PROVEN].** From a common-live
rolling state, at each indicated checkpoint the following cases partition the
possible continuation. The numbered checkpoint partitions are successive and
need not be mutually exclusive with one another.

1. **FirstStone placement.** `d_y=0` is a real-S terminal exit; `d_y=1` is
   the exact nonterminal first-unsafe exit; `d_y>=2` is first-safe.
2. **SecondStone placement.** `d_y=0` is a real-S terminal exit; every
   `d_y>=1` placement is nonwinning and passes the phase admission test.
3. **Old singleton debt.** If `E_S={e}` and `u=T(e)`, exactly one physical
   case holds on the unchanged live shadow board:

   - `u in X_Shat` (correct-role occupied);
   - `u in X_Fhat` (wrong-role occupied);
   - `u` is empty but not shadow-legal;
   - `u` is fresh, legal, and nonwinning for `Shat`; or
   - `u` is fresh, legal, and winning for `Shat`.

   Strict Definition 38.1 continues live only in the fourth case. The fifth
   is an actual physical `Shat` terminal stop, not an unresolved membership
   failure. The first three eject the strict handler.
4. **Completed nonwinning S pair.** `tau_E<=2` is exactly strict
   two-serviceability. `tau_E>2` ejects the trace before canonical service
   unless an outer physical reconciliation or sound stop changes the state.
5. **Paired F event.** The physical outcome table is:

   | real append | shadow append | strict disposition |
   |---|---|---|
   | nonterminal | nonterminal | live, subject to the next rolling clauses |
   | F-terminal | nonterminal | sound real-terminal closure; ET is vacuous |
   | F-terminal | `Fhat`-terminal | co-terminal aligned closure |
   | nonterminal | `Fhat`-terminal | misaligned raw closure; outside `A_FS2^ET` |

*Proof.* Items 1--2 are the integer form of the inherited deadline shield:
after a first placement `m=1`, so strict `delta>m` is `delta>=2`; after a
second `m=0`, it is exactly nonterminality. In item 3, physical occupancy is
disjoint and exhaustive; an empty coordinate is either in or outside the
current legal store, and a legal append either wins or does not. Thus
"certificate unavailable" means occupied or fresh-illegal, not a sixth case.
Item 4 is Definition 38.2(4) plus S29. Item 5 is Definition 55.1 and the
section-53 simultaneous final-event rule. QED.

When `E_S` is empty, absence of a filler is impossible on `Z^2`. Choose a
physical shadow stone of maximal q-coordinate. The distinct outward neighbors
`a+(1,0)` and `a+(1,-1)` have larger q, hence are empty, and both are legal at
distance one. At most one equals the forbidden `T(y)`. This is **[PROVEN]**
and removes filler nonexistence from the ejection list.

A real-S terminal exit in item 1 or 2 still owes P5R. Branch (C) covers it
only with an actual same-coupled-step `Shat` terminal certificate; under an
alleged-winning premise that certificate is itself the direct contradiction
of S51.1. A correct-role certificate or an F blocker may instead reconcile or
prevent the debt. Labels alone never change the physical verdict.

### 57.2 One newly covered ejection class

**Theorem S53 (correct-role occupied-certificate reconciliation) [PROVEN].**
At one-step rolling scope, suppose a common-live S-role checkpoint has
singleton debt `E_S={e}`, satisfies the inherited deadline shield, and

`u=T(e) in X_Shat`.

Strict `A_FS2` rejects the next required certificate because `u` is not fresh.
Before the next real-S query, physically recognize the already-present stone:
move `e` from `E_S` to `C_S`. This changes no occupancy or phase and makes the
old urgent family empty. For every subsequently observed legal nonterminal
`S@y` satisfying the ordinary phase guard--first-safe at `FirstStone`, or
nonwinning at `SecondStone`--the following exact subcases are available.

1. **Fresh/legal current image (branch A).** If `T(y)` is shadow-empty and
   shadow-legal, append `Shat@T(y)`. On a nonwinning append certify `y`
   immediately and set `E'_S=empty`. If the append wins, it is a physical
   shadow terminal stop; on a genuine history for an alleged-winning `sigma`,
   this is branch (C) and the direct counterstrategy contradiction.
2. **Filler rotation (branch B).** Otherwise append a fresh legal filler
   `Shat@f` with `f!=T(y)` and set `E'_S={y}`. If the filler is nonwinning, the
   inherited phase guard shields this one-step lag. Its future certificate
   availability remains an outer obligation. A terminal filler is merely a
   physical module stop unless the alleged-winning, `sigma`-consistent outer
   premise is present, in which case it is branch (C).

In either nonterminal subcase both histories reach the same actor and phase:
S at `SecondStone` after a `FirstStone` input, or F at `FirstStone` after a
`SecondStone` input.

*Proof.* Physical `Shat@T(e)` is exactly Definition 30.1's certificate, so the
pre-query reconciliation is physical recognition, not a label-only discount.
Doing it before `y` also avoids a transient two-debt microstate. Equivalently,
if reconciliation were delayed until after `y`, every old E-live `W` would
still satisfy

`delta'(W)-m'=(delta(W)-m)+1-1_{y in W}>0`,

so the ordering is not hiding a shield failure.

In subcase 1 the current image is by hypothesis a legal physical exact
certificate, leaving no lag. In subcase 2 use the two maximal-q outward
neighbors from the paragraph after S52; at least one is fresh, legal, and
different from `T(y)`. The only new E-live windows are through `y`. At
`FirstStone` the guard gives integer deficit at least two with `m=1`; at
`SecondStone` nonterminality gives deficit at least one with `m=0`. Thus the
new debt is shielded. One physical S-role stone is appended on each board, so
two nonwins give the stated matched phase. All old stones persist. QED.

This is a genuine membership-coverage advance. Correct-role occupied old
certificates are physically reconciled. The subfamily for which the current
image is fresh and legal continues by branch (A); the complement has only the
stated one-step branch-(B) rotation. Wrong-role occupancy cannot be
reconciled, because the persistent stone has the wrong owner, and a fresh but
shadow-illegal current target still lacks support.

**Corollary S53.1 (certified high-transversal reconciliation) [PROVEN].** At
singleton-debt F-checkpoint scope, if an F `FirstStone` checkpoint has
`E_S={e}`, possibly `tau_E>2`, but physical `Shat@T(e)` already exists, then
reclassifying `e` as certified makes `E_S` empty. The urgent family becomes
empty and the recomputed transversal number is zero.

This is the physical-reconciliation exception explicitly left open by S44;
it is not a certificate discount. The subsequent P3 event, future rolling
membership, and F-terminal alignment remain mandatory outer obligations.

### 57.3 What remains uncovered

At the exact scope of S52--S53, the following legal exits remain **OPEN**:

- nonterminal first-unsafe coordinates (`d_y=1`);
- real-S terminal coordinates without a physical same-step `Shat` certificate;
- wrong-role old-certificate occupancy;
- fresh but unsupported/illegal old certificates;
- `tau_E>2` without an already present correct-role certificate or other
  physical reconciliation; and
- shadow-`Fhat` terminal events outside `CAD+LOCK` or augmented exact
  admission.

Branches (A), (B), and (C) remain local alternatives at each S placement, but
none of these exits may drop P3, P5/P5R, persistence, service, or causality.

## 58. Per-pair `K=2`: an intra-pair rebinding cylinder

Round 6 proved that the support cut forces two episodes per S pair and that a
fixed isometry cannot close the colored second cut in S43.1's exact reactive
construction. The following is an honest finite construction in the surviving
direction, audited below against a fixed-map negative control.

**Theorem S54 (alternating-translation two-episode cylinder) [PROVEN].** At
the displayed finite scope, consider the genuine legal prefixes

```text
real:
F@(0,0);
S@(0,1),S@(1,1);
F@(1,0),F@(2,0).

shadow:
Shat@(0,0);
Fhat@(0,-1),Fhat@(1,-1);
Shat@(1,0),Shat@(2,0);
Fhat@(2,-1),Fhat@(3,-1).
```

The first five shadow placements are an S15 synchronization under

`T_1(q,r)=(q+1,r-1)`,

and the last shadow pair is one genuine legal `sigma` pair exactly matched to
the displayed real F pair. At the resulting common S `FirstStone` checkpoint,
two coordinate-reactive episodes admit the following alternating binding
execution.

Commit first to

`T_0(q,r)=(q,r-1)`.

Its physical complement consists of the proxies

`p_S=(2,0)` and `p_F=(3,-1)`.

Real S plays

`c_1=T_0^{-1}(p_S)=(2,1)`.

Append the fresh shadow filler `Shat@(3,0)` and rebind to `T_1`. At the reached
`SecondStone`, real S plays the now wrong-role cut

`c_2=T_1^{-1}(0,-1)=(-1,0)`.

Append `Shat@(-1,-1)` and rebind to `T_0`. The final total exact binding is
owner-faithful, both histories pass to the F role at `FirstStone`, and the
remaining physical proxies are

`Fhat@(3,-1)` and `Shat@(3,0)`.

*Proof.* Under the first `T_0` commitment, the images of the three real-F
stones are `(0,-1),(1,-1),(2,-1)` and the images of the two real-S stones are
`(0,0),(1,0)`, leaving exactly the displayed proxies. The coordinate
`c_1=(2,1)` is fresh and adjacent to `S@(1,1)`. The filler `(3,0)` is fresh
and adjacent to `Shat@(2,0)`. Under `T_1`, the three real-F stones map to
`(1,-1),(2,-1),(3,-1)` and all three real-S stones map to
`(1,0),(2,0),(3,0)`. The new proxies are therefore
`Fhat@(0,-1)` and `Shat@(0,0)`.

The coordinate `c_2=(-1,0)` is fresh, legal at real `SecondStone` (indeed it
is adjacent to the compulsory opener), and hits the wrong-role proxy under
`T_1`. Its filler `(-1,-1)` is fresh and within distance two of the shadow
opening (indeed adjacent to `Fhat@(0,-1)`). Under the restored `T_0`, all three
real-F stones and all four real-S
stones have the required physical owner images; only `(3,-1)` and `(3,0)`
remain unmatched. Every placement is radius-eight legal and no owner has six
stones, so both episodes and both rebindings are genuine and nonterminal.
QED.

This is not a universal candidate. The on-path `Fhat` moves extend by
least-legal choices to a total legal pure `sigma`, but that strategy is not
proved globally winning. No recurrence, arbitrary-S response, P5 theorem, or
universal P3 continuation is supplied. S54 proves only that changing `T`
inside the pair can move a persistent cut endpoint to the other side and can
close the exact two episodes on one physical cylinder.

**Fixed-map negative control.** This cylinder does **not** prove that either
rebind is necessary. At its initial `T_0` state, opposite-role proxy
`Fhat@(3,-1)` is adjacent to represented `Fhat@(2,-1)`, so S43.1's actual
first cut would be `(3,0)`, not the chosen `c_1`. Moreover, holding `T_0` fixed
maps `c_1=(2,1)` to the already present `Shat@(2,0)`, treats `Shat@(3,0)` as
the resulting proxy, and maps `c_2=(-1,0)` exactly to the appended
`Shat@(-1,-1)`. It reaches the same final proxies. Thus S54 proves consistency
of a two-rebinding execution only; it is not a cut adaptation, does not escape
S43.1's fixed-map subclass, and does not show that premise load-bearing. Full
`C_A^{K=2/pair}` remains **OPEN**.

## 59. Status and obligation ledgers

### 59.1 New theorem ledger

| Claim | Status | Exact scope |
|---|---|---|
| Definition 55.1 raw event closure | **Definition** | Live `A_FS2` handler plus physical Definition-46.1 F events, without assuming (46.4); simultaneous terminal closure retained |
| S46 debt update | **PROVEN** | Every legal paired F event and every selected opponent-unblocked physical window pair |
| Definition 55.2 `F-CAD_2^st` | **Definition** | Causal pre-query, many-window-to-one physical deficit portfolio; no point map; one-debt cover and terminal readiness only |
| Definition 55.3 `F-LOCK` | **Definition / residual obligation** | Canonical terminal event must hit an assigned window; separate from state maintenance and not derived by S46 |
| S47 conditional/augmented alignment | **PROVEN** | Canonical `CAD+LOCK` implies inherited ET; augmented least-choice handler chooses the ready window's unique hole, conditional on recurring state/nonterminal admission |
| Complete augmented terminal trace under least choice | **OPEN** | S41 proves canonical `CAD+LOCK` nonempty; no complete terminal execution of the distinct exact augmented selector is claimed |
| S48 S41/S42 invariant audit | **PROVEN** | Exact debt-one repayment on S41 and exact terminal-readiness failure on S42 |
| Canonical service maintains `F-CAD_2^st` on all `A_FS2` traces | **PROVEN (negation)** | S42 is first-safe, certificate-valid, `tau_E=0`, yet no real window is terminal-ready |
| S49 sixth-stone barrier | **PROVEN** | Every one-for-one post-S15 F-event selector, including every two-cell service augmentation; conditional on terminal age six |
| S49 alleged-winning fast history exists | **OPEN** | S42's strategy is legal, not proved globally winning |
| S50 adaptive earliest-cycle dichotomy | **PROVEN** | `Z^2`; every pure `sigma` and S15 synchronization; one causal first-safe/two-serviceable cycle protects the next certificate and reaches the fourth prescription |
| S50.1 earliest threat negative control | **PROVEN** | Five physical `Shat` stones have at most two immediate winning cells; their two-cell blocking response is nonwinning for `Fhat` |
| S51 finite-horizon stop trichotomy | **PROVEN** | Every alleged-winning `sigma`, fixed legal reachable nonterminal `sigma`-consistent raw checkpoint, and causal outer continuation; item 1 excludes the tested P5/(46.4) duty; uses S24 |
| S51.1 branch-(C) boundary | **PROVEN** | Common-live alleged-winning-`sigma` `Shat` nodes; a physical `Shat` win is already the counterstrategy contradiction |
| S52 membership partition | **PROVEN** | Strict rolling handler, legal real-S placements, old singleton certificate, completed pair, and paired F outcomes |
| S53 correct-role occupied reconciliation | **PROVEN** | Old `e` physically reconciled before the next query; branch A when current `T(y)` is fresh/legal, guarded one-step branch B otherwise |
| S53.1 certified high-transversal reconciliation | **PROVEN** | Singleton-debt scope: already-present correct-role certificate makes `E_S` and the urgent family empty |
| S54 alternating-translation cylinder | **PROVEN** | One finite legal intra-pair `T_0/T_1/T_0` two-episode execution; legal `sigma`, not proved winning |
| Full per-pair `K=2` success class | **OPEN** | S54 is not a total response algorithm and supplies no recurrence or terminal theorem |
| Universal alignment for every alleged-winning `sigma` | **OPEN** | Neither universal `F-CAD_2^st` plus canonical `F-LOCK` nor an own-winning-history fast terminal is proved |
| Global P0--P6 plus P5R coupling | **OPEN** | The branch system does not cover every legal real-S continuation |
| `NL_F` | **OPEN** | D2 remains the bridge; neither determinacy alternative is selected |

There are no **SKETCH** or **CONJECTURE** theorem claims in this round. The
surviving K=2 datum is a **PROVEN finite execution**, not a sketched global
candidate or a separation from fixed `T_0`.

### 59.2 Round-6 review's authoritative twelve obstacles

| # | Authoritative obstacle | Round-7 disposition |
|---:|---|---|
| 1 | Full per-pair and broader zero-lag branch (A) | **FINITE CONSISTENCY EXAMPLE; OPEN.** S54 gives one legal two-rebinding execution, but fixed `T_0` also closes it and its first coordinate is not S43.1's forced cut. It proves no separation. Arbitrary-S coverage, recurrence, P3/P5, total nonisometric point recodings, non-total/window recodings, and indefinite repair remain open. |
| 2 | Pre-checkpoint and recurring P3 coverage | **PROVEN on inherited `A_FS2^ET`; PARTIAL raw advance; OPEN globally.** S50 reaches the next two actual prescriptions of every fixed `sigma` through one guarded cycle. It does not maintain all later membership or terminal alignment. |
| 3 | Coverage outside `A_FS2` | **ONE NEW CLASS COVERED; OPEN otherwise.** S52 gives successive checkpoint partitions. S53 physically reconciles correct-role occupied old certificates; its fresh/legal-current-image subfamily continues by branch A and its complement has only a guarded one-step branch B. S53.1 covers a high-transversal singleton debt when that certificate already exists. Wrong-role, unsupported, first-unsafe, terminal, and uncertified `tau_E>2` exits remain open. |
| 4 | P5R through every lag/recode | **PARTIAL.** S53 preserves the inherited deadline shield in its branch-B rotation. A filler win is branch C only on a genuine alleged-winning, `sigma`-consistent outer history; otherwise it is a module stop. S14 and S25 remain mandatory. |
| 5 | Canonical F-service compatibility | **SUFFICIENT STATE TEST FOUND; OPEN universally.** `F-CAD_2^st` is the physical deficit invariant and `F-LOCK` is the separate canonical selector duty. S41 has both; S42 has no ready state portfolio. S50 proves ordinary E-transversal serviceability for one cycle only--not CAD or lock feasibility. No theorem reaches the S24 horizon. |
| 6 | Universal shadow-`Fhat` terminal fidelity | **CONDITIONAL POSITIVE; NEGATIVE-AT-FAST-SCOPE; OPEN globally.** S47 transfers canonical alignment from state plus lock and proves the augmented terminal choice at admitted states. S49 makes terminal age six impossible for every one-for-one selector. No alleged-winning `sigma` is forced to have that age. |
| 7 | Reverse legality for spatial carriers | **UNCHANGED OPEN for spatial carriers.** The event and deficit constructions never invert `z`; S18/S13 and sequential unsupported/collision sets remain binding on inverse/FIFO proposals. |
| 8 | Strategy domain and physical persistence | **PROVEN on new finite scopes; OPEN globally.** S47 uses reached legal prescriptions, S50's counterplay is a function of fixed pure `sigma`, S53 uses an actual correct-role stone, and S54 keeps every filler/proxy. No physical stone is removed or recolored. |
| 9 | Global causality | **PARTIAL.** `svc_{E+CAD}` fixes its portfolio before the query and chooses its event cell after the reached prescription; S50 uses deterministic pure-strategy look-ahead through its own fixed next shadow append. Other branches and future backing plans still owe S12. |
| 10 | Universal window-certificate maintenance | **NEW F-ROLE MODULE; OPEN globally.** S46 gives the exact debt update. S47 transfers terminal meaning only with state maintenance plus canonical lock, or with the admitted augmented rule; it does not prove recurring maintenance. Arbitrary S turns and simultaneous P2/P3/P5R remain open. |
| 11 | High-transversal service and permanent fencing | **ONE CERTIFIED EXCEPTION; OPEN generally.** S53.1 reconciles a singleton high-transversal debt only when its correct-role physical shadow certificate already exists. S30's exact `tau_E=5`, S31's six-blocker cost, availability, interrupted installation, and P3 compatibility remain binding. |
| 12 | Strategy-specific reachability and outcome | **PARTIAL.** S50 is adaptive on every fixed `sigma` and reaches its fourth prescription, but does not force it to win. S49 is negative if it does. S51 reduces the general route to membership plus real-stop avoidance. No arbitrary alleged-winning strategy is refuted; `NL_F` remains open. |

### 59.3 Round-4 review's ten-item agenda

| Agenda item | Round-7 status | Exact advance and remaining duty |
|---:|---|---|
| 1. Pre-checkpoint P3 transfer | **PROVEN AT S40 SCOPE; PARTIAL beyond it** | S50 crosses one more actual `sigma` cycle with canonical service. Universal recurrence and terminal fidelity remain open. |
| 2. P2/P4 at each real-S coordinate | **PARTIAL** | S53 covers one occupied-certificate ejection. S54 executes two displayed zero-lag coordinate responses with changing `T`, but the same cylinder is fixed-`T_0` solvable and is not a cut adaptation. First-unsafe, wrong-role, unsupported, and general adaptive coordinates remain open. |
| 3. P5R during every lag/recode | **PROVEN in inherited guarded/certified classes and S53's admitted one-step continuations; OPEN globally** | The new debt is admitted by the exact phase guard. S14 remains binding on unguarded literal lag; real-terminal exits still need physical branch C or another P5R mechanism. |
| 4. F-service compatibility | **CONDITIONAL AT CERTIFIED SCOPE; OPEN globally** | `F-CAD_2^st` is a sufficient readiness certificate; canonical service separately owes `F-LOCK`, while augmented exact service chooses the ready hole. S50 proves only `tau_E<=2` in the first adaptive cycle. S49 excludes a sixth-stone terminal. |
| 5. Permanent-fence installation | **OPEN** | S31 is unchanged. S53.1 uses a pre-existing shadow certificate, not a newly installed six-blocker fence. |
| 6. Reverse P3 legality | **PROVEN irrelevant for event pairing; OPEN for spatial transfer** | No new inverse claim is made. S18/S13 remain mandatory for every spatial/FIFO carrier. |
| 7. Shadow-`Fhat` terminal fidelity | **CONDITIONAL on `CAD+LOCK` / admitted augmented scope; OPEN universally** | S47 is the strict-scope transfer, S49 the exact count barrier, and S42 the legal diagnostic. First- and second-placement terminal windows are both included. |
| 8. Strategy domain and persistence | **PROVEN on displayed/new classes; OPEN globally** | Every append in S50, S53, and S54 is physical. Rebinding changes only representation; all old stones stay in every rule calculation. |
| 9. Causality | **PROVEN locally; OPEN globally** | Service is selected after S; the second prescription remains sequential. S50's adversary may compute fixed future pure-strategy values, as S12 permits, but no future F coordinate is announced to S. |
| 10. Strategy-specific reachability and outcome | **PARTIAL** | S50 replaces a special legal `sigma` by every fixed strategy for the earliest cycle. Its fast terminal is a dichotomy branch, not a forced behavior. S51 states the remaining finite-horizon task. |

### 59.4 P0--P6/P5R cross-ledger

| Obligation | Status after round 7 | Binding disposition |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **PROVEN on all new finite traces; OPEN globally** | S47 queries only genuine reached histories. S50's `z_1,z_2` are the actual values of the fixed pure strategy; S54's on-path strategy is legal but not alleged-winning. |
| `P1 OPENING/CADENCE` | **PROVEN for S15 and all paired finite modules** | Every nonwinning first append reaches `SecondStone`; every nonwinning second passes control. A terminal first placement suppresses the second on that board, and the coupled trace closes by its stop rule. |
| `P2 REAL->SHADOW` | **PARTIAL** | S53 supplies exact zero-lag copying on its branch-A subfamily and a guarded filler queue on branch B; S54 supplies a two-episode point-binding cylinder. Universal coordinate coverage is open. |
| `P3 SHADOW->REAL` | **PROVEN AT inherited `A_FS2^ET`; CONDITIONAL on raw `CAD+LOCK`/augmented scope; OPEN globally** | Temporal pairing remains legal without inverse coordinates. Terminal alignment and continued serviceability are strict-class duties. |
| `P4 COLLISION` | **PARTIAL** | Correct-role old-certificate occupancy is now covered. Wrong-role occupancy remains physical and cannot be relabeled. S54 is only a legal alternating labeling; fixed `T_0` handles the same coordinates, so it proves no new collision class. |
| `P5 SHADOW-F-TERMINAL` | **PROVEN on `CAD+LOCK` and admitted augmented traces; OPEN globally** | S47 transfers alignment at those strict scopes; state maintenance alone is insufficient, and S49 forbids alignment at terminal age six. S20 remains binding elsewhere. |
| `P5R REAL-S-TERMINAL-REFLECTION` | **PROVEN in inherited classes and S53's admitted one-step continuations; OPEN globally** | S53's branch-B rotation preserves `delta>m` locally; a filler-terminal result is branch C only under the genuine alleged-winning outer premise. S14/S25 remain mandatory. |
| `P6 CAUSALITY` | **PROVEN for the new local selectors; OPEN globally** | No service cell is fixed across an intervening S turn. Pure-strategy counterplay computation in S50 is not an exposed future real-F prescription. |

## 60. Hostile-review attack surface and regression matrix

### 60.1 Load-bearing limitations

1. **Raw and aligned classes are different.** A failed terminal event is an
   extension of `A_FS2^EV`, never a member of `A_FS2^ET`.
2. **State and selector are separate.** `F-CAD_2^st` does not imply where a
   fixed canonical service cell lands. S47 part 1 also assumes `F-LOCK`; part 2
   chooses the ready hole under the distinct augmented rule. Neither part
   proves recurring portfolio existence after every adversarial S turn.
3. **One-debt is not zero-debt.** Immediately after S15 and one straight
   `sigma` pair, a shadow deficit-two window may correspond to a real
   deficit-three window. Requiring debt zero there would make the positive
   class artificially empty at the first relevant checkpoint.
4. **Terminal readiness is the red line.** Debt one is allowed at shadow
   deficit two but forbidden at deficit one. S42 fails at exactly this
   transition.
5. **Many-to-one windows are not merged stones.** The portfolio may assign
   several shadow obligations to one real window. No noninjective point map,
   recoloring, or virtual occupancy is inferred.
6. **The augmented selector has a feasibility test.** At a ready terminal
   state its unique-hole choice is proved legal. At a nonterminal event it
   still needs a service-compatible cell and a next portfolio; no universal
   existence theorem or complete terminal trace is claimed. S49 shows that the
   ready state itself cannot exist at the earliest terminal age.
7. **The count barrier is conditional on behavior, not on legality alone.**
   It applies to an alleged-winning `sigma` if that strategy wins with stone
   six on the constructed history. No theorem forces that behavior.
8. **S42 remains diagnostic for the alleged-winning-only question.** It does
   refute universal-over-legal canonical maintenance and physically realizes
   the S49 barrier.
9. **S50 is one cycle, not an induction.** First-safety and `tau_E<=2` follow
   from the small owner counts. At later cycles accumulated S geometry can
   create first-unsafe or high-transversal exits.
10. **S50's look-ahead uses a fixed pure strategy.** The actual carrier still
    queries sequentially. The construction would not apply to a hidden or
    randomized strategy without a separate argument.
11. **S50.1 is a negative control, not a defense theorem.** It only rules out
    the simplest earliest double-threat coercion; it supplies no globally
    winning `Fhat` response.
12. **S51 is a reduction, not a controller.** Avoiding membership failure and
    real-F service wins through the finite horizon is the unproved work.
13. **Branch C is a contradiction stop.** On an alleged-winning history a
    physical `Shat` win cannot be used and then ignored; it already refutes
    `sigma`.
14. **S53 needs the correct physical owner.** `Shat@T(e)` is a certificate;
    `Fhat@T(e)` is a blocker/collision and cannot be fixed by labels.
15. **S53.1 does not solve arbitrary `tau_E>2`.** It removes the debt only
    when its correct-role certificate already exists.
16. **S54 is one consistency cylinder.** It proves that two rebindings can be
    physically legal, not that they are needed: fixed `T_0` also closes it,
    and the first coordinate is not S43.1's forced cut. It supplies no
    totality, recurrence, winning-strategy reachability, or terminal fidelity.
17. **Second placements remain sequential.** Every post-first deficit,
    service set, certificate, and terminal test is recomputed on the actual
    two physical boards.
18. **Every physical stone persists.** No proof erases, moves, recolors, or
    undoes a proxy, filler, service cell, or old certificate.
19. **Branches are locally alternative and globally conjunctive.** Choosing
    A, B, or C never waives P3, service, persistence, P5/P5R, causality, or
    the regression suite.
20. **No outcome inflation.** A strict-subclass carrier and a conditional
    negative barrier do not select a determinacy alternative. `NL_F` is open.

### 60.2 Binding regression matrix

| Regression | Round-7 treatment | Remaining boundary |
|---|---|---|
| S12 preannounced real-F coordinate | Canonical/augmented service is selected after the S turn and used before S moves again; S50 computes opponent-strategy values, not an exposed future F placement | Every outer repair/backing plan still owes S12 |
| S13 fixed-isometry FIFO frontier | F-role event pairing is neither inverse copying nor FIFO; S54's changing labeling is legal but has a fixed-`T_0` realization too | Every proposal satisfying S13's fixed-map FIFO premises remains excluded; S54 proves no escape |
| S14 literal one-cell terminal lag | S50 and S53 retain the deadline shield; terminal real S remains an explicit exit | Unguarded literal lag remains excluded |
| S18 proxy-supported reverse illegality | No F-role inverse is used; S41 remains the exact illegal-inverse stress inside the canonical certified class | Spatial P3 proposals still owe reverse legality |
| S20 proxy-fabricated `Fhat` win | S47 transfers it under `CAD+LOCK` or the admitted augmented rule; S49 proves a count regime where readiness is impossible | State maintenance plus the canonical lock remains open universally |
| S25 older-surplus real-S win | S53 physically certifies the old debt before relabeling and preserves `delta>m` | Other lag/recode branches still owe terminal memory |
| S30 exact `tau_E=5` fork | S53.1 covers only the special case with an already-present correct-role certificate | Candidate-own reachability, ordinary reconciliation, and five-versus-two service remain open |
| S31 six-blocker permanent fence | No installation theorem is claimed | Availability, interruption, S occupation, and P3 compatibility remain binding |

## 61. Objective dispositions and sharp resume point

### 61.1 Alignment verdict

**PARTIAL -- CONDITIONAL POSITIVE-AT-STRICT-SUBCLASS.** `F-CAD_2^st` is a
causal physical F-window state invariant with the exact update (55.6). For
canonical service, `F-LOCK` remains a separate selector duty; S47
proves the conditional transfer but does not derive that duty. The augmented
exact rule genuinely chooses the ready real hole at a terminal prescription,
conditional on recurring state and nonterminal-service admission. S41 proves
the canonical `CAD+LOCK` class nonempty. S42 is the state-invariant negative
audit and S49 is the selector-independent sixth-stone barrier. A complete
terminal trace for the exact least-choice augmented rule is not separately
proved nonempty.

The adaptive negative route is also **PARTIAL**. S50 reaches the sharp fast
test on every fixed strategy's own history, but does not force the fourth
prescription to win. S51 proves that continued membership plus real-stop
avoidance would force a later misalignment within the winning strategy's
finite horizon. Only branch-(C) coercion is circular/directly refutational;
the general nonterminal counterplay route is not.

### 61.2 Membership outcome

**ONE NEW EJECTION CLASS COVERED.** Correct-role occupied old-debt targets are
strict-`A_FS2` freshness exits but are physically reconciled by S53. When the
current image is fresh/legal the continuation is genuine branch A; otherwise
only a guarded one-step branch-B rotation is proved. The same physical
certificate reconciles a singleton `tau_E>2` debt at an F checkpoint.
Wrong-role occupancy, unsupported certificates, first-unsafe/terminal S
coordinates, and general high-transversal states remain open.

### 61.3 Per-pair `K=2`

**FINITE CONSTRUCTION ONLY.** S54 is an exact legal two-episode intra-pair
rebinding execution, but fixed `T_0` closes the same cylinder and its first
coordinate is not S43.1's forced cut. It proves neither cut adaptation nor a
total per-pair carrier.

### 61.4 Most valuable theorem and next question

**Most valuable new theorem:** S49, because it proves the selector-independent
physical count boundary at which every one-for-one two-event alignment rule
must fail. S46 supplies the exact state-update equation, while S47 is the
honest conditional/augmented transfer theorem.

**Sharpest next question [OPEN].** For every alleged-winning `sigma`, can a
causal controller extend S50's one-cycle construction to the S24 horizon while
maintaining `F-CAD_2^st` and, for canonical service, forcing `F-LOCK` (or
reaching a sound real-F stop), while repairing every certificate collision and
keeping every S debt first-safe and two-serviceable; or must some own-history
event hit the S49 barrier or another physically misaligned terminal window?

## 62. Provenance

### 62.1 Repository state

**Requested input state.** Branch `hunt/gap-raw`, input commit
`09e27a93` (the round-6 hostile review and binding section 53). This authoring
pass creates no commit and does not amend, reset, or move a branch reference.

During the session a read-only check observed the branch reference at
`a8a0b92d`, an external descendant of `09e27a93`. A name-only comparison
showed only an unrelated prompt and `GAP_RAW_PROOF_ROUND7.md` /
`GAP_RAW_REVIEW_ROUND7.md`. Those files were not opened or used as evidence.
The required strategy-stealing corpus and the six production rule files were
unchanged across that name-only difference. Pre-existing untracked entries
were left untouched.

The only file created by this pass is `STRATEGY_STEALING_ROUND7.md`.

### 62.2 Required corpus read first, in order and in full

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including binding section 35, then
   `STRATEGY_STEALING_REVIEW_ROUND4.md`;
5. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, then
   `STRATEGY_STEALING_REVIEW_ROUND5.md`; and
6. `STRATEGY_STEALING_ROUND6.md`, including binding section 53 and the
   terminal-closure definition, then `STRATEGY_STEALING_REVIEW_ROUND6.md`
   and its authoritative updated obstacle list.

No `GAP_RAW_*` document was read as mathematical evidence.

### 62.3 Rule sources read in full

The following production files were read in full after the required corpus:

- `packages/hexo_engine/rust/src/coord.rs`;
- `packages/hexo_engine/rust/src/legal.rs`;
- `packages/hexo_engine/rust/src/rules.rs`;
- `packages/hexo_engine/rust/src/board.rs`;
- `packages/hexo_engine/rust/src/state.rs`; and
- `packages/hexo_engine/rust/src/tactics.rs`.

The proofs use only the rooted opening, physical radius-eight support,
sequential board insertion, per-placement terminal detection, terminal
no-continuation rule, three six-window axes, and append-only forward history
tied down in section 54.2.

### 62.4 Machine and mutation boundary

No Cargo command, Lean build, harness, executable search, or proof-search
program was run. No production source, prior proof artifact, or unrelated
workspace entry was edited. All coordinate, count, cadence, deficit, window,
and transversal arguments are hand proofs.
