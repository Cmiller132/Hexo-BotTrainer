# Strategy stealing in engine Hexo, round 8: the alignment dichotomy

**Worktree:** `hunt/gap-raw` at requested input commit `175ca45e`  
**Date:** 2026-07-18  
**Maintenance-horn verdict:** **POSITIVE ON A NONEMPTY RESERVE/QUIET CYCLE
CLASS, AND NEGATIVE AT AN EXACT CLIFF** -- portfolio-aware prepayment and
reserve placement maintain the augmented lock on their admitted cycles, while
a first-event hit at scalar deficits `(mu_H,mu_R)=(2,3)` defeats every
one-for-one selector before the mandatory second query.  
**Forcing-horn verdict:** **POSITIVE-AT-A-NAMED FAST-WINNER CLASS** -- every
alleged-winning strategy of uniform post-S15 shadow depth at most eight is
forced, on its own S50 history, to win with its sixth `Fhat` stone and hence
to meet S49 misalignment.  
**Ejection verdict:** **ONE CONDITIONAL FIRST-UNSAFE STOP CLASS COVERED** -- a
mirror-clean first-unsafe event supplies a physical `Shat` completion on the
second coupled step, independently of real S's second coordinate; the class
is a direct-refutation stop criterion, not a continuing alleged-winning
trace.  
**Global target:** `NL_F` remains **OPEN**.

The main new maintenance fact is a scalar collapse hidden inside the
many-window-to-one definition of `F-CAD_2^st`. At a live F-role query, the
existence of a portfolio depends only on the smallest physical real-F and
shadow-`Fhat` deficits. If the shadow minimum is two and the real minimum is
three, CAD is admitted at debt one. But if the reached *first* prescription
hits a shadow deficit-two window, the shadow minimum becomes one while one
paired real placement can reduce the real minimum only to two. No reassignment,
least-choice rule, or stronger service selector can restore terminal readiness
before `sigma` must be queried at `SecondStone`. The exact alternatives are
prepayment before the query, an unmatched real append that breaks event
cadence/P3, or exit from the common-live branch. This strictly extends S48's
canonical-service negation to every one-for-one portfolio-aware selector at
the named physical boundary.

The complementary positive class prepays exactly one event earlier. A
dynamically quiet `(2,3)` pair uses its first real placement to reduce the
real minimum to two and its second to complete singleton E service. Thereafter
the reserve-one handler catches an aging event whenever the minima are
two/two, and an avoiding rolling S pair preserves the designated reserve.
The modified S41 cylinder realizes quiet ingress, a second-event catch, one
complete rolling cycle, and an augmented co-terminal unique-hole stop. This
is a complete nonempty trace for the stronger explicit handler, although its
strategy remains diagnostic rather than alleged-winning.

The negative horn is sharpened without using S51 circularly. From an S15
checkpoint, the next six shadow placements have owners

`Fhat,Fhat,Shat,Shat,Fhat,Fhat`.

No owner can win during the first five by stone count. If an alleged-winning
strategy does not win at placement six on every counterplay, a legal
nonwinning `Shat` pair extends one compatible branch through placements seven
and eight. Thus the least uniform S24 depth is either six or at least nine.
S50 already constructs a guarded raw history through placement six for every
fixed strategy. Hence the entire depth-at-most-eight class is forced onto
S49's six-versus-five terminal mismatch on its own legal history.

Finally, the new ejection module treats a real first-unsafe move whose
four-stone precursor window has a clean physical shadow image. Exact copying
of that first coordinate gives `Shat` five stones in the image window. On the
associated second step the handler fills the remaining shadow hole, whatever
legal second coordinate real S selected. This is an actual branch-(C)
counterstrategy stop under an alleged-winning premise, not a label change or
a continued terminal history. Indeed the clean shadow window already makes
the alleged-winning premise contradictory at the pre-move checkpoint; the
folded round-4 `C_shield` extension gives a reachable nonempty cylinder only
for a legal diagnostic strategy, not a globally winning one.

All results are hand proofs. No Cargo command, Lean build, harness, executable
search, or proof-search program was run. No `GAP_RAW_*` document was opened or
used as mathematical evidence. This authoring pass creates no commit.

## 64. Statement boundary and binding inherited state

### 64.1 Roles, target, and status discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real second
player. In the role-swapped shadow, `Shat` represents real S and is the
opener, while `Fhat` represents real F and follows a fixed pure second-player
strategy `sigma`. The target remains

`NL_F : exists pure sigma_F, for every pure sigma_S, S never wins`.

Here `S never wins` permits a finite F win or an infinite nonterminal history
with neither owner completing six. Round-2 Theorem D2 is inherited
**[PROVEN from the CITED Gale--Stewart open-determinacy theorem]** for the
declared unbounded-board macro-game:

`NL_F <=> S has no winning strategy`.                         (D2)

This round does not refute every alleged-winning `sigma`; D2 therefore does
not yield `NL_F`. Every named or load-bearing assertion is marked
**PROVEN**, **SKETCH**, **CONJECTURE**, **OPEN**, or **CITED**. Definitions
are stipulative. Objective labels such as **NEGATIVE-AT-SCOPE** and
**POSITIVE-AT-CLASS** are not proof statuses. There are no machine-verified
claims.

### 64.2 Production rules used [PROVEN]

On axial coordinates the engine distance is

`d((q,r),(q',r'))=max(|q-q'|,|r-r'|,|(q-q')+(r-r')|)`.

The only opening is `F@(0,0)`. Along nonterminal forward play the cadence is

`F ; S,S ; F,F ; S,S ; F,F ; ...`.

A normal placement is legal exactly when its coordinate is physically empty
and lies in the color-blind radius-eight legal store. A nonwinning first
placement is inserted before the second is validated. A Q-, R-, or QR-axis
length-six window is checked after each single append and before phase
advancement. A winning first placement suppresses the second; a terminal
state exposes no legal continuation.

These facts are implemented in
`packages/hexo_engine/rust/src/coord.rs:1-4,9-20,76-95`,
`legal.rs:17-18,123-145`, `rules.rs:11-44`, `board.rs:83-105`,
`state.rs:149-160,203-252,265-273,289-357`, and
`tactics.rs:13-17,21-75,205-208,451-485`. Forward placements insert physical
occupancy and history. The public apply/undo path is an analysis/MCTS
facility, not a legal-history operation. Every proxy, filler, queued stone,
certificate, and service placement therefore remains physically present and
continues to affect support, blocking, and terminal windows.

The executable carrier is `i16`. General results retain the inherited `Z^2`
idealization. The explicit ejection witness is already bounded by the folded
round-4 proof; no new large-coordinate gadget is introduced.

### 64.3 Folded errata and authoritative open state

Sections 35, 44, 53, and 63 are binding in full. In particular:

- branches (A), (B), and (C) are alternatives only for one observed real-S
  coordinate; P0--P6, P5R, service, persistence, causality, and the regression
  suite remain conjunctive in every branch;
- every negative construction must occur on the candidate's own legal,
  `sigma`-consistent history when it is used against a strategy-specific
  candidate; a merely legal labeled state is diagnostic only;
- S13 remains binding on fixed-isometry FIFO schedules and S14 on unguarded
  literal one-cell lag; the rolling queue avoids neither theorem outside its
  expressly guarded and physically reconciled scopes;
- a queue rotation has a transient two-debt physical microstate, and every
  reconciliation or filler append that wins is a physical `Shat` stop; a
  P3-shaped legal trace is not positive compatibility with every alleged-
  winning strategy;
- the terminal closure makes the final paired F event atomic across the two
  boards, but never authorizes continuation of either terminal engine state;
- total nonisometric zero-lag point recodings remain a separate open class;
  common-only real wins and simultaneous legality plus P2/P3/P5/P5R terminal
  maintenance remain explicit duties;
- S51 is a finite-horizon **non-disjoint stop cover**, not a partition or a
  controller. Voluntary truncation belongs to its membership/continuation
  failure item, and P5/(46.4) is deliberately excluded from that item;
- S51.1 is exact: a physical `Shat` win on a genuine alleged-winning history
  is already the counterstrategy contradiction and may not be reused as a
  continuing branch;
- a portfolio policy is fixed before the trace and is totalized with
  `Option Portfolio`; `None` is a strict-class exit, not an admissible
  portfolio; and
- physical histories are append-only. Rebinding changes representation only;
  no old physical stone is erased, moved, recolored, or undone.

The round-7 hostile review's twelve-item unresolved-obstacle list is the
authoritative input ledger. Sections 69--70 carry every item, the older
ten-item agenda, the P0--P6/P5R cross-ledger, and all restored caveats
forward explicitly.

### 64.4 Round-8 quantifier boundary

The maintenance theorem below concerns a live common F-role **first**
microstep followed immediately, after two nonwins, by the mandatory shadow
`SecondStone` query. It does not infer an obligation at the end of an F pair,
where an intervening S/`Shat` turn may block or replace near windows before
the next F query.

The forcing theorem concerns the named fast class of alleged-winning
strategies. Existence of such a strategy is **OPEN**. Its conclusion is a
conditional strategy-own misalignment theorem, not a claim that an arbitrary
alleged winner is fast.

The constructive first-unsafe stop theorem covers only mirror-clean events.
S62 separately gives a general common-phase negative stop cover and a
conditional `tau_E>2` handoff barrier; neither supplies a terminal coordinate
outside the mirror-clean class. First-unsafe events with a missing image, an
`Fhat`-blocked image window, or an illegal first image therefore remain in the
authoritative open list as positive-coverage problems.

## 65. The maintenance horn: feasibility, reserve cycles, and the readiness cliff

### 65.1 Scalarization of the causal portfolio

At a live common pre-query F-role microstep use the inherited families
`U_R^F,U_H^F` and deficits (55.1)--(55.4), and put

`mu_R=min{delta_R^F(W):W in U_R^F}`,                         (65.1)

`mu_H=min{delta_H^F(V):V in U_H^F}`.                         (65.2)

Both minima exist and lie in `{1,...,6}`. The physical position is finite,
there are unblocked far-away windows of deficit six, and liveness excludes
deficit zero. Fix once and for all well-orders of physical windows and legal
coordinates. When a least-deficit real window is needed below, take the first
one in this fixed order. This makes every selection a function of the reached
prefix rather than of an unqueried prescription.

**Lemma S55 (scalar CAD feasibility) [PROVEN].** At every such live prefix an
`F-CAD_2^st` portfolio exists exactly when

`mu_H>=3`, or `mu_H=2 and mu_R<=3`, or `mu_H=1 and mu_R=1`.   (65.3)

Moreover, whenever it exists, a causal portfolio is obtained by assigning
every shadow window of deficit at most two to one least real window attaining
`mu_R`.

*Proof.* If `mu_H>=3`, the portfolio domain is empty. If `mu_H=2`, every
member of its domain has shadow deficit two. The one-debt inequality permits
a real deficit of at most three, so an assignment exists exactly when
`mu_R<=3`. If `mu_H=1`, terminal readiness applied to a minimum shadow window
requires a real window of deficit at most one. Liveness makes that deficit
exactly one. Conversely, one real deficit-one window can serve every shadow
window of deficit one or two, satisfying readiness for the former and the
one-debt bound for the latter. Many-to-one assignment is expressly permitted
by Definition 55.2. All inputs to the least choice are physical prefix data,
so special-casing these choices and returning `None` elsewhere totalizes one
pre-trace `Option Portfolio` policy exactly as required by section 63. QED.

This lemma is only an existence test for the state certificate. It does not
turn the inherited canonical coordinate into a locked coordinate and does not
remove any E-debt service duty.

### 65.2 The one-placement readiness/service intersection

Consider a common-live F `FirstStone` pre-query state with
`(mu_H,mu_R)=(2,2)`. Let the reached legal prescription `z` be nonterminal and
hit at least one shadow deficit-two window. Thus, before the mandatory second
query, its shadow append changes the minimum to one. Define

`C_R = union{W\X_F : W in U_R^F and delta_R^F(W)=2}`.        (65.4)

Every member of `C_R` is real-legal: it is an empty hole of a six-window
already containing four real-F stones, and is at line distance at most two
from one of them.

For a legal first real coordinate `k`, let the residual urgent family consist
of the pre-turn E-urgent windows whose physical hole sets were not hit by
`k`. Write `k in S_E` when, after a nonwinning `F@k`, one further legal
real-F coordinate can hit every residual urgent hole set and itself remain
nonterminal. A terminal second coordinate is kept separate as a sound real
stop; it is not called a nonterminal service continuation.

**Lemma S56 (exact catch/service trade-off) [PROVEN].** Under the preceding
hypotheses:

1. the post-event prefix admits `F-CAD_2^st` before the second query exactly
   when `k in C_R`;
2. it admits both that portfolio and a two-nonterminal-placement completion
   of the inherited E service exactly when `k in C_R intersect S_E`; and
3. if the pre-turn urgent family has a legal singleton transversal `s`, then
   every `k in C_R` either leaves `s` as the legal second service cell, has
   already hit the service when `k=s`, or is followed by a sound real-F stop.

*Proof.* Every real unblocked window starts with deficit at least two and a
single F append lowers the deficit of precisely the windows containing its
coordinate. Hence the post-event real minimum is one exactly for
`k in C_R`. The shadow minimum is one, so S55 proves item 1. The remaining
placement completes the deadline service without terminating exactly when
the residual urgent family has the stated nonterminal one-cell transversal,
which is the definition of `S_E`; this proves item 2. For item 3, if `k!=s`,
the old singleton remains empty, legal, and hits every residual urgent
window. If `k=s`, service is already complete. If the chosen follow-up wins,
the carrier closes at the sound real stop and owes no later service. QED.

Thus `tau_E<=1` removes the service-versus-catch conflict at the prepaid
boundary. For `tau_E>1`, `C_R\S_E` is an exact negative-control region for a
*nonterminal* continuation, although filling a newly created real one-hole
window on the second placement may still give a sound real-F stop.

### 65.3 A nonempty quiet prepayment class

Call an actual two-prescription F pair **dynamically portfolio-quiet** when,
at each of its two reached pre-query shadow states, the actual prescription
misses every then-current shadow deficit-two window. This condition is
sequential. Merely missing the pair-start family is insufficient, since the
first prescription can create a new deficit-two window for the second to age.

**Theorem S57 (quiet one-debt prepayment) [PROVEN].** Suppose a common-live F
`FirstStone` checkpoint has

`(mu_H,mu_R)=(2,3)` and `tau_E<=1`,                          (65.5)

and its actual `sigma` pair is dynamically portfolio-quiet. There is a causal
portfolio-aware real handler which either reaches a sound real-F stop, or
completes both paired events nonterminally, performs every mandatory E
service, maintains `F-CAD_2^st` before the second query, and ends with

`mu_H=2 and mu_R<=2`.                                       (65.6)

Any real-F win instead is a sound earlier stop.

*Proof.* Query the first actual prescription. It cannot win because the
shadow minimum is two, and quietness leaves every current minimum window at
deficit two. Choose a hole `c` of a least real deficit-three window. Such a
hole is legal: among three occupied cells of a length-six window, every hole
is at line distance at most three from one of them. Append `F@c`; this makes
the real minimum at most two. S55 therefore supplies the next portfolio
before the second query. If the urgent family has singleton transversal `s`,
use `s` on the second real event unless `c=s`, in which case service is
already complete and a legal padding cell is used. If `tau_E=0` (empty
urgent family), use the fixed least legal filler, and if it wins, close at
the sound real-F stop (review round-8 Finding 4). Dynamic quietness again
keeps the shadow minimum at two, while an F append cannot increase the real
minimum. If the service or padding cell wins, close soundly; otherwise (65.6)
holds. Every choice follows its reached prescription, and the second
prescription is queried only after both first appends are known nonterminal.
QED.

The class is physically nonempty beyond S41's terminal trace. Follow S41
through rolling S pair 1. Replace its next real/shadow F pair by

`F@(3,0),F@(0,5) / Fhat@(2,5),Fhat@(3,5)`.

At its start the S41 windows

`W={(0,0),...,(5,0)}`, `V={(1,0),...,(6,0)}`

have real/shadow deficits three/two, and the urgent family has singleton
service coordinate `(0,5)`. Both displayed prescriptions are dynamically
quiet; `(3,0)` is the prepayment and `(0,5)` is the service. S41 rolling pair
2, unchanged, then reaches a common live checkpoint with
`(mu_H,mu_R)=(2,2)` and `tau_E=0`. All cited legality, certificate, and
nonterminal checks are the folded S41 checks; the changed coordinates are
adjacent to already physical support. Extending the displayed prescriptions
by a least-legal rule gives a total pure strategy, but it is a legal
diagnostic strategy and is **not** claimed alleged-winning.

### 65.4 The exact readiness cliff

**Theorem S58 (first-event one-debt readiness cliff) [PROVEN].** Fix a
common-live F-role **first** microstep with

`(mu_H,mu_R)=(2,3)`.                                        (65.7)

Suppose the actual legal prescription `z_1` hits a shadow deficit-two window.
For every legal one-for-one paired real placement `k_1`, both appends are
nonterminal and the reached common `SecondStone` prefix admits no
`F-CAD_2^st` portfolio. Consequently neither canonical service, the S47
least-choice handler, portfolio reassignment, nor any stronger query-first,
common-phase, one-event-per-microstep one-for-one
selector can maintain `F-CAD_2^st`--and hence cannot maintain
`CAD+LOCK`--before it must query `z_2` (review round-8 Finding 5: the
theorem is not a negative about asynchronous or prepayment architectures
that leave S40's one-event-per-microstep carrier).

*Proof.* The shadow append changes the minimum from two to one and cannot yet
complete six. Every real unblocked window begins with deficit at least three,
and one real append lowers any such deficit by at most one. Thus the reached
real minimum is at least two; in particular the real append also cannot win.
The `mu_H=1` line of S55 requires `mu_R=1`, so no portfolio exists at the
reached pre-query prefix. This argument quantifies over every legal `k_1` and
does not depend on a canonical order, a chosen assignment, or E urgency. QED.

The alternatives are exact. Prepaying to `mu_R<=2` before the aging event
reaches the S56 boundary. If that prepayment is absent, the carrier must
strictly exit, decline the mandatory genuine second query, or insert a second
unmatched real append before that query. The last option may be an engine-
legal real second placement, but it destroys S40's common-phase,
one-event-per-microstep P3 carrier; it is not a repair inside the admitted
class. This is the first-microstep theorem only. After a second-microstep
failure an intervening `Shat` turn can block the one-hole shadow window, so no
permanent failure at the next F checkpoint is inferred.

S58 strengthens S48. S42 is still the concrete legal own-history diagnostic,
but its `sigma_dagger` remains unproved winning. The theorem itself is
selector-independent and applies on any alleged-winning strategy's own
history if that history reaches (65.7).

### 65.5 Reserve-one rolling cycles

Define the prefix condition

`RES_1 : (mu_H=1 => mu_R=1) and (mu_H>=2 => mu_R<=2)`.       (65.8)

It implies S55. Fix the causal portfolio `Pi_min` which assigns every near
shadow window to the least real window attaining `mu_R`. Define an augmented
lock `F-LOCK^+` to mean the same reached-event incidence (55.7), but for the
coordinate selected by the new handler rather than for Definition 55.3's
independently fixed canonical coordinate. This notation does **not** relabel
the handler as canonical `F-LOCK`.

The reserve handler acts sequentially as follows. It evaluates `Pi_min`
before each query. A terminal prescription is paired with the unique hole of
the assigned real deficit-one window. If the real minimum is already one at
a nonterminal prescription, that unique hole is always an available sound
real stop; on an admitted continuing branch the handler may instead use the
still-required singleton service cell or its fixed legal filler, neither of
which can increase the real minimum.
At `(mu_H,mu_R)=(2,2)`, an aging first prescription is caught in `C_R`; the
next real placement may take the resulting unique hole for a sound stop, or,
when a nonterminal continuation is required, S56 schedules the singleton E
service. Quiet events use the first still-free singleton service cell or a
legal filler. After every live F pair, designate `W_*` to be the fixed-order
least real F-unblocked window attaining the post-pair `mu_R`; under `RES_1`
its deficit is at most two. A second-event catch is allowed to leave this
reserve at deficit one.

One **reserve-one rolling cycle** is a genuine common F-`FirstStone` to common
F-`FirstStone` nonterminal segment satisfying all of these conditions:

1. its entrance satisfies `RES_1`, `tau_E<=1`, and all inherited live
   `A_FS2` clauses;
2. its F pair is generated by the reserve handler and both paired events are
   nonterminal;
3. the following real-S/`Shat` pair is causal, first-safe, certificate-fresh,
   service-admissible, and nonterminal under the inherited rolling rules;
4. both real-S coordinates avoid the designated post-F reserve window `W_*`;
   and
5. the exit has `tau_E<=1` and is common-live.

Call the class of finite concatenations (with their sound terminal closures)
`R_1(sigma)`. Two semantics are explicit (review round-8 Finding 6): the
reserve handler REPLACES the inherited canonical F-service choice in the
"inherited live `A_FS2`" clauses, and "sound terminal closures" means §53's
atomic paired-final-event convention — both associated physical appends
occur inside the one coupled closing event.
Fix the continue/stop choice causally by a prefix predicate as
part of the handler before play (the finite witness prefixes below are
special-cased, followed elsewhere by the fixed least choices). If a stated
candidate or rolling clause is absent, the analysis certificate returns
`None` and the trace exits the strict class, while the underlying pure move
rule still chooses the fixed least legal coordinate. Thus the class is
generated by one totalized pure rule, not by a retrospective choice.

**Theorem S59 (reserve-one maintenance to any admitted horizon) [PROVEN at conditional cycle scope].**
For every pure `sigma`, every finite
concatenation in `R_1(sigma)` maintains `F-CAD_2^st` at every reached F query,
passes `F-LOCK^+` at every shadow-terminal event, completes its singleton E
service on every continuing turn, and otherwise closes at a sound real-F
stop. In particular this holds for every admitted prefix up to the S24
horizon of every alleged-winning `sigma`. Universal membership of those
strategies in `R_1` is not claimed.

*Proof.* S55 makes `Pi_min` admissible under (65.8). A nonterminal
prescription at shadow minimum one misses every deficit-one shadow window,
so that minimum persists; a real F append cannot increase its minimum.
At minima two/two, a first-event aging transition is exactly S56, and
`tau_E<=1` schedules the remaining second-placement service or a sound stop.
If aging occurs instead on the second event, the quiet first event has already
paid any singleton service; choosing a hole of a real deficit-two window
directly changes the two minima to one/one. All other nonterminal F events
preserve (65.8) monotonically. At a terminal
prescription, readiness makes the assigned real window deficit one, so its
unique legal hole gives the S47(2) co-terminal append and `F-LOCK^+`.

Across the following S-role pair, `Shat` stones can only block and remove
shadow F-windows; they cannot lower `mu_H`. Real S stones can block real
F-windows, but avoidance leaves `W_*` unblocked at its old deficit, so the
next real minimum is no larger than that reserve. Hence (65.8) holds at the
next entrance. The remaining admission clauses are hypotheses 3--5. Induct
over the finite concatenation. S24 is used only to select a finite prefix of
an already admitted concatenation, never to prove membership or suppress an
intermediate stop. QED.

This rolling class is physically nonempty, exercises a second-event catch,
and has a complete augmented co-terminal closure. Use the S57 witness through
unchanged S41 rolling pair 2. Keep S41's actual shadow order and pair

`Fhat@(-8,0),Fhat@(5,0) / F@(1,5),F@(4,0)`.

The first event is quiet and off `V/W`. The second changes `V/W` from
deficits two/two to one/one. Then use unchanged S41 rolling pair 3,

`S@(8,0),S@(8,1) / Shat@(4,4),Shat@(10,0)`.

The real S cells avoid `W`; the final debt `(8,1)` has `tau_E=0`; neither
shadow cell blocks `V`. Thus the exit has `mu_H=mu_R=1`. At the next
`FirstStone` event pair S41's terminal prescription and the reserve hole,

`Fhat@(6,0) / F@(5,0)`.

For this totalized witness rule, put `W={(0,0),...,(5,0)}` before the other
real deficit-one q-window in the fixed window order. Then `Pi_min` assigns the
terminal shadow `V` to this `W`, and the displayed real coordinate is its
unique hole. This fills `V/W` co-terminally and witnesses `F-LOCK^+`. Every
new support is at distance at most eight, all cells are fresh, and no owner
has six before the final event; the remaining checks are S41's accepted
physical checks.
The chronology uses the same legal `sigma_star` prescriptions as S41, though
the real service cells are portfolio-aware. As before, `sigma_star` is not
claimed alleged-winning.

The maintenance-horn answer is therefore two-sided and exact at named scope:
S57--S59 give a nonempty positive portfolio-aware class beyond S41, while
S58 proves that no augmented one-for-one policy can cross the unprepaid
`(2,3)` aging cliff. Canonical `F-LOCK`, arbitrary S-pair admission, and
membership through the horizon for every alleged winner remain **OPEN**.

## 66. The forcing horn: a post-S15 horizon gap

### 66.1 Fast-winner depth

Fix an alleged-winning pure shadow second-player strategy `sigma` and a legal
reachable S15 synchronization `h`. Let `d_sigma(h)` be the least integer `n`
such that, against every complete/maximal legal `Shat` counterplay from `h`,
`Fhat` has won within `n` further physical shadow single placements.
Equivalently, it bounds every branch of the compatible nonterminal-prefix
tree; voluntarily truncated finite prefixes are not counterplays in this
definition. S24 makes this integer finite by its finitely-branching-tree
argument at the fixed checkpoint.

Define the named checkpoint class

`FAST_8^{S15}={(sigma,h): sigma is alleged-winning, h is a reachable S15 synchronization for sigma, and d_sigma(h)<=8}`. (66.1)

The ownership sequence after `h`, with placement numbers local to `h`, is

| local placement | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---|---|---|---|---|---|---|---|
| owner | `Fhat` | `Fhat` | `Shat` | `Shat` | `Fhat` | `Fhat` | `Shat` | `Shat` |
| owner count after append | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 |

The last row uses the S15 counts `(|X_Fhat|,|X_Shat|)=(2,3)` and displays the
count of the owner who just moved.

**Lemma S60 (post-S15 horizon gap) [PROVEN].** For every alleged-winning
`sigma` and reachable S15 checkpoint `h`,

`d_sigma(h)=6 or d_sigma(h)>=9`.                             (66.2)

*Proof.* Neither owner can have six stones during the first five local
placements. If `Fhat` wins on local placement six against every legal
intervening `Shat` pair, the least uniform depth is six. Otherwise choose a
legal continuation on which the fourth post-S15 `Fhat` prescription is
nonterminal. The resulting state is live at `Shat FirstStone`. A legal
`Shat` placement exists. If such a placement won, it — followed by any
off-path totalization of the counterstrategy — would itself be a legal
counterplay directly contradicting the alleged-winning premise; that direct
shadow-game contradiction needs no coupled-node premise (review round-8
Finding 7: S51.1 is formally stated at a genuine common-live coupled node,
which this arbitrary shadow branch has not been shown to possess, so the
direct argument replaces the S51.1 citation here). Hence under the
premise choose a nonwinning first placement. The same reasoning at
`SecondStone` supplies a nonwinning second placement. A compatible
nonterminal branch therefore exists through local placement eight, so the
uniform depth is at least nine. QED.

The two hypothetical placements seven and eight are used only to prove the
gap. No branch-(C) history is continued. This is the exact point at which
S51.1 helps without turning S51's finite stop cover into a controller.

### 66.2 Forced sixth-stone misalignment for the fast class

**Theorem S61 (fast-winner sixth-stone forcing) [PROVEN at conditional class scope].**
For every `(sigma,h) in FAST_8^{S15}`, S50's causal legal real-S
continuation on `sigma`'s own genuine history reaches its fourth post-S15
prescription and has all of the following properties:

1. the rolling step passes first safety, supplies its fresh old-debt
   certificate, ends with `tau_E<=2`, and remains a genuine raw event history;
2. the fourth prescription wins and is `Fhat`'s sixth physical stone;
3. the paired real F has exactly five physical stones, and no earlier real-F
   sound stop occurred; and
4. the event is S51 stop-cover item 3 / Definition 55.1 outcome 4: shadow
   terminal and real nonterminal.

The sixth-stone count and terminal conclusions in items 2--4 are independent
of the choice, order, or portfolio-aware augmentation of the two one-for-one
real service coordinates, provided they are legal/nonterminal until the
tested event and no unmatched real-F append is made. Item 1 is specifically
S50's admission guarantee for its supplied carrier; it is not asserted for
an arbitrary replacement service trace.

*Proof.* S50 supplies the named causal continuation through local placement
six for every fixed pure strategy. S60 changes `d_sigma(h)<=8` into
`d_sigma(h)=6`. Because `d_sigma` is defined over complete/maximal
counterplays, note that if S50's finite prefix were nonterminal after local
placement six, it could be extended legally to a complete/maximal
counterplay, contradicting `d_sigma(h)=6` (review round-8 Finding 8); hence
the reached fourth prescription must win on this
particular legal counterplay. The resulting event satisfies S51 cover
item 3, possibly alongside another cover item (S51 is a non-disjoint
cover). At root: `B_sigma=11` gives `d_sigma(h) <= B_sigma - 5 = 6` for
every reached S15 continuation, validating the `FAST_<14` to `FAST_8^{S15}`
bridge. Paired F events change the real/shadow F-role
counts from `1/2` at S15 to `3/4` and then `5/6`. The first three post-S15
events cannot win for real F by count, and the final paired append still
leaves only five real-F stones. S49 now gives selector-independent terminal
misalignment. S50 supplies all item-1 admission facts on this same
`sigma`-consistent history. QED.

This theorem does not extrapolate S50 coverage to the whole S24 horizon. It
uses S50 only through the already-proved sixth local placement; S60 is what
turns that prefix into a forced terminal event for the named class.

There is a corresponding root-level strategy class. Let `B_sigma` be the least
uniform physical shadow placement index, counted from the shadow opener, by
which alleged-winning `sigma` has won. The same cadence/count argument gives

`B_sigma=11 or B_sigma>=14`.                                 (66.3)

Thus every alleged-winning `sigma` with `B_sigma<14` has `B_sigma=11`, lies
in the S61 horn at every reached S15 checkpoint, and is forced onto the
sixth-stone misalignment along S50's continuation. Call this strategy class
`FAST_<14`. Its nonemptiness is **OPEN**. S24 proves only that each alleged
winner has some finite bound; it gives neither `B_sigma<14` nor any universal
numerical bound. Alleged winners outside this horn satisfy
`d_sigma(h)>=9` at each checkpoint where the fast premise fails.

The forcing-horn verdict is therefore **POSITIVE for `FAST_8^{S15}` and
`FAST_<14`**, but **OPEN for the slow class**. This is a meaningful split of
the S51 controller problem, not a proof of `NL_F`.

## 67. Ejection coverage: deadline stops and a mirror-clean first-unsafe class

### 67.1 The common-phase deadline stop cover

Let `C_CP^S` denote causal common-phase S-role candidate handlers which,
for each observed actual legal real-S append, make exactly one associated
actual legal `Shat` append. Two nonwins preserve the same
`FirstStone`/`SecondStone` phase. A physical `Shat` win is the S51.1
contradiction stop. Such a candidate succeeds at a real-S win only with the
required physical same-coupled-step shadow terminal transfer--P5R when the
winning window meets represented debt, and the restored common-only real-win
duty otherwise. No unmatched F placement or phase-lag escape belongs to this
class.

At a real-S pre-placement state let the **S deficit** of an F-unblocked window
be

`delta_R^S(W)=6-|W intersect X_S|`.                          (67.1)

**Theorem S62 (deadline-deficit stop cover) [PROVEN].** Fix an
alleged-winning `sigma`, a reachable genuine common S/`Shat` microstep, and an
F-unblocked real window `W`. Let `m=2` at `FirstStone` and `m=1` at
`SecondStone`. If

`1<=delta_R^S(W)<=m`,                                       (67.2)

then every handler in `C_CP^S` meets the following exhaustive, generally
non-disjoint stop cover on a causal legal real-S continuation:

1. an associated `Shat` append wins, which directly contradicts the
   alleged-winning premise and stops under S51.1; or
2. real S becomes terminal within the remaining `m` placements while the
   associated shadow append is nonterminal, so the applicable physical
   terminal-reflection duty fails.

*Proof.* Choose real S's remaining coordinates successively from the holes of
`W`. There are at most `m`. Every chosen hole is legal: `W` already contains
at least four real-S stones, so a hole is at line distance at most two from
one of them. If a real cross-window wins earlier, item 2 is reached earlier
unless the associated shadow append supplies item 1. Otherwise filling the
last hole completes `W`. After each observed real append, inspect the one
physical associated `Shat` append. Its win is item 1 and is never continued.
If none wins, the shadow remains nonterminal when real S completes six, so
the appropriate P5R or common-only terminal-transfer duty fails. QED.

This is a stop criterion and negative control, not a live repair. It does not
identify a shadow winning coordinate by itself.

**Corollary S62.1 (first-unsafe no-live-repair boundary) [PROVEN].** Let an
observed legal real first placement `S@y` have S52 value `d_y=1`, witnessed by
an F-unblocked window whose post-`y` unique hole is `x`. If the associated
first `Shat` append wins, stop by S51.1. If it is nonterminal, the boards are
at common `SecondStone` and S62 applies with `m=1`. Hence no
`C_CP^S` branch-(A) or branch-(B) response can continue live through this
first-unsafe event: its next response either physically wins for `Shat` or
fails the real-S terminal-reflection duty when real S chooses `x`.

The coordinate `y` is an observed choice on the actual candidate-own coupled
history, and `x` is chosen only after that reached prefix. This is not a
free-standing labeled gadget.

### 67.2 Conditional high-transversal handoff

**Corollary S62.2 (`tau_E>2` common-handoff barrier) [PROVEN at named handler scope].**
At an actual common F `FirstStone` checkpoint with `tau_E>2`, let a
one-for-one real/shadow F pair be completed without either board becoming
terminal and hand control to a genuine common S/`Shat FirstStone` state. Then
some pre-turn urgent E-window `W` was missed by both real F placements, stays
F-unblocked, and has `delta_R^S(W)<=2`. The next `C_CP^S` turn is covered by
S62.

*Proof.* The two real F coordinates form a set of size at most two. Since the
urgent-hole transversal number exceeds two, that set misses the hole set of
some `W in U_E`. Neither append therefore blocks `W`; S-role occupancy did
not change during the F turn, so its deficit remains at most two. Apply S62
at the common handoff. QED.

Thus an attempted high-transversal continuation has an exact set of exits:
a sound real-F stop during service; the inherited aligned or misaligned
shadow-terminal event; loss of common-phase membership; a physical `Shat`
contradiction stop on the next turn; or failure of the real-S terminal
transfer. This strengthens S44/S44.1 only for the named actual nonwinning
common-phase handoff. S45's `tau_E=5` S30 position remains an abstract stress
case without candidate-own alleged-winning reachability, and S31's permanent
fence installation remains **OPEN**.

### 67.3 Constructive mirror-clean stop

**Definition 67.1 (`A_FU^mc`, mirror-clean first-unsafe stop class).** At a
genuine common-live S/`Shat FirstStone` prefix consistent with a fixed pure
`sigma`, observe a legal real append `S@y` with `d_y=1`. Let `W` witness this
and let `x` be its post-`y` unique hole. For the reached translation/D6
certificate isometry `T`, require

- every cell of `W\{y,x}` to have an actual physical
  `Shat@T(c)` on the shadow board; and
- `T(y)` and `T(x)` to be shadow-empty.

No `E_S=empty` premise is imposed; unrelated physical debt may remain. The
six cells `T[W]` consist of four `Shat` stones and two holes, so a separate
`Fhat`-unblocked premise would be redundant.

**Theorem S63 (mirror-clean first-unsafe terminal stop) [PROVEN].** Every
prefix in `A_FU^mc` has a causal two-step branch-(C) handler. Append the
associated physical `Shat@T(y)`. If it wins, stop immediately. Otherwise,
after any actual legal real second coordinate `r`, append
`Shat@T(x)` on the associated final coupled step. This second shadow append
is legal and completes `T[W]`, independently of `r`. If `S@r` is real
terminal it supplies its physical terminal reflection; if `S@r` is
nonterminal, the `Shat` win directly refutes alleged-winning `sigma`.

*Proof.* An isometry maps the engine window `W` to an engine window `T[W]`.
Before the first shadow append it has four physical `Shat` stones and the two
fresh holes `T(y),T(x)`. Either hole lies within line distance at most two of
one of those stones and is therefore shadow-legal. Filling `T(y)` leaves a
five-stone window unless a cross-window has already won. In the nonwinning
case both boards reach `SecondStone`; `T(x)` is still fresh, legal, and the
unique hole of `T[W]`. Its append physically completes six. The inherited S26
terminal-supply and S52 same-coupled-step P5R semantics permit this associated
shadow certificate if the real second append has just terminated, but neither
terminal engine history is continued afterward. QED.

There is an important logical boundary. Already at the pre-`y` checkpoint,
`T[W]` is a shadow deficit-two window on `Shat`'s current turn. By the S39
terminal-supply argument and S51.1, an alleged-winning `sigma` cannot actually
survive the two-hole counterplay. Accordingly `A_FU^mc` is a
**direct-refutation stop criterion**, not a nonterminal alleged-winning
continuation class. Its physical nonemptiness is witnessed only against a
legal diagnostic strategy.

For that witness, use the folded round-4 `C_shield` terminating extension
immediately before its final real-S pair, with

`T(q,r)=(q-2,r)`,

`S@(q,1), q=0,1,2,3`, and `Shat@(q,1), q=-2,-1,0,1`.

Take `W={(q,1):q=0,...,5}`, `y=(4,1)`, and `x=(5,1)`. Then
`T(y)=(2,1)` and `T(x)=(3,1)` are fresh. The folded extension proves all
prefix legality and no earlier win; its two final coupled placements complete
the real and shadow windows. The unrelated old debt at `(0,5)` remains
physical and shielded, demonstrating why the definition need not assume an
empty debt set. The on-path strategy is not claimed globally winning.

The ejection outcome is therefore a genuine positive stop for the
mirror-clean first-unsafe class, an exact no-live-repair theorem for every
other common-phase one-append handler at the same deadline, and a conditional
extension to `tau_E>2` handoffs. Missing/blocked image cells, wrong-role and
unsupported certificates, phase-lag handlers, and arbitrary high-transversal
reconciliation remain **OPEN**.

## 68. Alignment-dichotomy synthesis

The new local decision boundary is:

| Reached condition | Round-8 disposition | Scope caveat |
|---|---|---|
| `mu_H>=3` | CAD portfolio is vacuous | Service and rolling admission remain separate |
| `mu_H=2, mu_R<=2` (one event) | S55/S56 one-event CAD/catch feasibility | Local feasibility only |
| `mu_H=2, mu_R<=2` (rolling) | Reserve handler maintains CAD | Conditional on EVERY `R_1` admission clause (RES_1; inherited live `A_FS2` conditions; handler-generated nonterminal F pair; causal first-safe certificate-fresh service-admissible nonterminal S pair; both real-S cells avoid `W_*`; common-live exit with `tau_E<=1`) |
| `mu_H=2, mu_R=3`, dynamically quiet pair | S57 prepays to the reserve class | Conditional on a common-live F `FirstStone` checkpoint with `tau_E<=1` AND both actual reached prescriptions being dynamically quiet |
| `mu_H=2, mu_R=3`, first prescription hits a deficit-two window | S58: every one-for-one selector fails CAD before query 2 | An unmatched real append leaves the S40 carrier |
| `mu_H=1, mu_R=1` | Augmented unique-hole rule aligns a shadow terminal, or can stop by a real win | This is augmented `F-LOCK^+`, not universal canonical `F-LOCK` |
| `d_sigma(h)<=8` | S60 forces `d=6`; S61 forces S49 misalignment | Named alleged-winner class may be empty |
| `d_sigma(h)>=9` | Slow-horizon controller problem remains | S24 supplies only finiteness |
| mirror-clean first-unsafe | S63 supplies a physical `Shat` terminal stop | Direct-refutation class, not live alleged-winning continuation |
| other common-phase deadline deficit | S62 gives contradiction-or-transfer-failure cover | Phase lag and other physical certificates are outside the theorem |

Accordingly the two horns meet at, but do not close, a sharp gap. The positive
maintenance policy needs the real reserve no later than the event at which a
shadow deficit-two window ages to one. The fast forcing theorem produces a
terminal age where the global one-stone count offset makes such a reserve
impossible under one-for-one cadence. (REWRITTEN per review round-8
Finding 1, which REFUTED the original synthesis claim:) A universal
fast-or-reserve theorem would connect two local interface results, but would
not by itself prove `NL_F`. The fast S49 branch still needs an outcome-level
argument — S61 refutes the tested one-for-one terminal-transfer carrier, it
does not refute `sigma` or decide the game — and all outer-coverage
obligations listed in §69 remain binding: the unresolved region contains at
least the fast S49 outcome branch, slow winners, the complement of the
quiet/`R_1` admission class, and the unresolved ejection and outer-carrier
classes. This section is a PARTIAL INTERFACE MAP, not an alignment dichotomy
or exhaustive synthesis.

### 68.1 New theorem ledger

| Result | Status | Exact contribution |
|---|---|---|
| S55 scalar CAD feasibility | **PROVEN** | Collapses the many-to-one portfolio-existence question to `(mu_H,mu_R)` |
| S56 catch/service intersection | **PROVEN** | Gives the exact one-placement readiness condition and separates nonterminal service from a sound stop |
| S57 quiet one-debt prepayment | **PROVEN on a named class** | Pays the `(2,3)` debt while completing `tau_E<=1`; folded S41 supplies a legal witness |
| S58 first-event readiness cliff | **PROVEN** | Extends S48 from canonical service to every one-for-one portfolio-aware selector |
| S59 reserve-one rolling maintenance | **PROVEN at conditional cycle scope** | Maintains CAD plus augmented lock on every admitted concatenation and has a complete physical co-terminal witness |
| S60 post-S15 horizon gap | **PROVEN** | An alleged winner's local uniform depth is six or at least nine |
| S61 fast-winner forcing | **PROVEN at conditional strategy-class scope** | Forces every `FAST_8^{S15}`/`FAST_<14` member onto S49 on its own S50 history |
| S62 deadline-deficit cover | **PROVEN** | Excludes a live common-phase one-append repair and yields the conditional `tau_E>2` handoff barrier |
| S63 mirror-clean first-unsafe stop | **PROVEN** | Supplies an explicit causal physical branch-(C) completion; folded `C_shield` witnesses diagnostic nonemptiness |
| `NL_F` | **OPEN** | Slow alleged winners and universal outer membership remain uncovered |

## 69. Binding status and obligation ledgers

### 69.1 Round-7 review's authoritative twelve obstacles

The obstacle descriptions below preserve the review-confirmed open state;
the disposition column records only the incremental round-8 change.

| # | Authoritative obstacle | Round-8 disposition |
|---:|---|---|
| 1 | **Full per-pair and broader zero-lag branch (A).** S54 is one `T_0/T_1/T_0` execution with a fixed-`T_0` realization; arbitrary-S recurrence, intra-pair changing isometries, total nonisometric point recodings, non-total/window recodings, and indefinite one-repair-per-placement remain. | **OPEN, unchanged.** S57--S59 are F-role event policies and use the inherited rolling S handler. S63 uses one fixed reached isometry for a terminal stop; it proves no universal recoding carrier. |
| 2 | **Pre-checkpoint and recurring P3 coverage.** One genuine history must reach all later prescriptions with common phase, serviceability, and terminal rules. | **PARTIAL ADVANCE; OPEN globally.** S59 inducts through every *admitted* `R_1` concatenation and its complete witness goes beyond S41. Arbitrary S pairs need not satisfy its reserve-avoidance or `tau_E<=1` clauses. S50/S61 still reach only the fourth post-S15 prescription outside that class. |
| 3 | **Coverage outside strict `A_FS2`.** First-unsafe, unreflected real-S terminals, wrong-role occupancy, unsupported certificates, uncertified `tau_E>2`, and Fhat-terminal events outside admitted alignment remain. | **ONE POSITIVE STOP PLUS ONE NEGATIVE COVER; OPEN otherwise.** S63 covers mirror-clean first-unsafe by a physical direct-refutation stop. S62.1 proves no other live common-phase one-append repair at that deadline. Missing/blocked/illegal images, unreflected terminals, wrong-role and unsupported certificates, phase lag, and arbitrary high-transversal exits remain. |
| 4 | **P5R through every lag and recode.** Debt-meeting real wins need shielding/certification/F blocking/same-step Shat terminal; common-only real wins require an outer physical transfer. S14 and S25 bind. | **PARTIAL.** S63 supplies the physical same-step terminal for its mirror-clean class. S62 distinguishes P5R from the restored common-only-win duty and proves the exact failure if neither is supplied. All other lag/recode paths, S14, S25, and common-only physical transfer remain open. |
| 5 | **Canonical and augmented F-service compatibility.** CAD state and canonical LOCK are separate; nonterminal candidates and portfolios must recur through the horizon. | **EXACT CONDITIONAL POSITIVE/NEGATIVE SPLIT.** S57--S59 prove a stronger explicit reserve handler on a nonempty `tau_E<=1` cycle class and a complete augmented terminal trace. S58 proves every one-for-one policy fails at the unprepaid `(2,3)` aging cliff. Universal canonical `F-LOCK`, arbitrary service, and universal horizon membership remain open. |
| 6 | **Universal shadow-Fhat terminal fidelity.** Every later first/second terminal prescription needs a same-event real certificate or a strategy-own misalignment theorem. | **FAST ONE-FOR-ONE TERMINAL TRANSFER FORCED TO FAIL; OUTCOME OPEN; OPEN slowly.** S61 forces strategy-own sixth-stone misalignment for `FAST_8^{S15}` and `FAST_<14` — this defeats the tested carrier, not `sigma` itself; the fast outcome branch needs a separate outcome-level argument. S59 aligns every terminal reached inside its reserve class. Later terminals of slow alleged winners outside that class remain open. |
| 7 | **Reverse legality for spatial carriers.** Every inverse/FIFO scheme owes S18, S13, and updated unsupported/collision sets. | **OPEN, unchanged.** Temporal reserve pairing uses no inverse. S63's forward image is legal by its explicit physical window; it supplies no reverse carrier. |
| 8 | **Strategy domain and physical persistence.** Every event must lie on one genuine legal append-only history agreeing with total `sigma`; all old stones retain every rule effect. | **PROVEN on new local traces; OPEN globally.** S57/S59 witnesses and S63's folded witness are physical and totalizable but their strategies are explicitly diagnostic, not alleged-winning. S61 uses S50 on each candidate's own actual prescriptions. Nothing is erased or relabeled physically. |
| 9 | **Global causality.** Future backing, recoding, and repair choices may not expose a future real-F coordinate across an S turn. | **PROVEN locally; OPEN globally.** Portfolios are selected before queries, event coordinates only after reached prescriptions, and `W_*` only before the immediately following observed S pair. S50's fixed-pure-strategy computation remains within S12. Universal outer repair is absent. |
| 10 | **Universal window-certificate maintenance.** New windows, reassignment after arbitrary S turns, canonical LOCK, common-only real wins, and simultaneous legality plus P2/P3/P5/P5R terminal maintenance require one recurring physical handler. | **SCALARIZED AND CONDITIONALLY MAINTAINED; OPEN universally.** S55 gives exact existence, S59 handles reserve-avoiding cycles, and S58 gives the failure cliff. Arbitrary S-created windows and common-only real wins remain duties. In particular, “simultaneous P2/P3/P5R” does **not** discharge simultaneous legality, P5, or the common-win obligation. |
| 11 | **High-transversal service and permanent fencing.** S30 has exact `tau_E=5`; S31 costs six blockers; availability, interruption, S occupation, reconciliation, and P3 compatibility remain. | **CONDITIONAL NEGATIVE EXTENSION; OPEN positively.** S62.2 shows that any actual nonwinning two-stone service at `tau_E>2` misses an urgent window and enters the deadline stop cover at a common handoff. S53.1's already-certified singleton remains the only positive exception. No fence installation or general reconciliation is supplied. |
| 12 | **Strategy-specific reachability and outcome.** A controller must preserve membership and avoid a real sound stop through each strategy-dependent horizon; otherwise universal coupling and `NL_F` stay open. | **FAST/SLOW SPLIT; OPEN globally.** S61 resolves every member of the named fast class on its own S50 history. Nonemptiness of that class is open, and S24 gives no fast bound. Slow alleged winners still need the recurring outer controller; `NL_F` remains open. |

### 69.2 Round-4 review's ten-item agenda

| Agenda item | Status after round 8 | Exact advance and remaining duty |
|---:|---|---|
| 1. Pre-checkpoint P3 transfer | **PROVEN on inherited S40 and admitted `R_1`; PARTIAL beyond** | S59 gives induction over every admitted reserve-one concatenation and a complete co-terminal trace. S61 uses S50 only through the next actual pair. Universal recurrence is open. |
| 2. P2/P4 at each real-S coordinate | **PARTIAL** | S63 handles the mirror-clean first-unsafe coordinate by an explicit image-terminal response; S62.1 gives the no-live-repair boundary. Wrong-role, unsupported, missing-image, and general adaptive coordinates remain. **S13's fixed-isometry one-stone FIFO frontier failure remains binding in this row**; the new terminal image completion is not FIFO recurrence. |
| 3. P5R during every lag/recode | **PROVEN in inherited classes and `A_FU^mc`; OPEN globally** | S63 is a physical same-step terminal supply. S62 retains the distinct common-only real-win duty. **S14 remains binding** on unguarded literal lag, and S25 binds older surplus. |
| 4. F-service compatibility | **PROVEN on `R_1`/quiet `tau_E<=1`; NEGATIVE at S58; OPEN globally** | S56 gives the exact catch/service intersection, S57 prepays, and S59 rolls/terminates. Canonical LOCK and `tau_E>1` nonterminal service remain unresolved. |
| 5. Permanent-fence installation | **OPEN** | S62.2 is a stop cover after missed service, not an installed blocker. S31's six-cell cost and all availability/interruption/P3 duties remain. |
| 6. Reverse P3 legality | **PROVEN irrelevant for temporal event pairing; OPEN for spatial transfer** | No F-role inverse is used. S18 and S13 remain mandatory for inverse/FIFO proposals; S63 proves only forward-image legality in its terminal window. |
| 7. Shadow-Fhat terminal fidelity | **PROVEN on admitted reserve traces; FORCED FAILURE on fast class; OPEN universally** | S59 gives augmented co-terminal lock. S61 makes the sixth-stone failure unavoidable for fast alleged winners under one-for-one cadence. Slow, non-reserve terminals remain. |
| 8. Strategy domain and persistence | **PROVEN on displayed/new finite classes; OPEN globally** | S61 is candidate-own and `sigma`-consistent. The positive witnesses use total legal strategies not claimed winning. All appends persist; rebinding changes no board. |
| 9. Causality | **PROVEN locally; OPEN globally** | Dynamic quietness is tested only after each reached query; `Pi_min` is pre-query; the second prescription stays sequential. No coordinate is exposed across an intervening S turn. |
| 10. Strategy-specific reachability and outcome | **PARTIAL** | S61 proves a conditional forcing theorem for `FAST_<14`; S59 proves a conditional maintenance theorem. Fast-class nonemptiness and the slow-strategy controller remain open. |

### 69.3 P0--P6/P5R cross-ledger

| Obligation | Status after round 8 | Binding disposition |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **PROVEN on all new finite traces; OPEN globally** | S57/S59 query only reached prescriptions and totalize their policies before play; their nonemptiness strategies are legal but not alleged-winning. S61 uses S50's genuine history for each fixed alleged winner. S63 is a direct-refutation stop and never continues a won shadow state. |
| `P1 OPENING/CADENCE` | **PROVEN for all new paired modules** | First and second events are sequential; terminal first placements suppress seconds; terminal closure adds only the associated same-microstep append. S58 explicitly identifies an unmatched extra real placement as outside, not inside, the common-phase carrier. |
| `P2 REAL->SHADOW` | **PARTIAL** | S63 provides exact forward images for one mirror-clean first-unsafe pair. S53's occupied-certificate handler and S54's finite cylinder persist. Universal coordinate coverage is open and S13 still binds FIFO. |
| `P3 SHADOW->REAL` | **PROVEN on inherited `A_FS2^ET` and conditional `R_1`; OPEN globally** | S59 pairs every reached prescription with a legal reserve/service coordinate and proves augmented terminal incidence. S58 proves exact failure of CAD admission at one one-for-one boundary. No inverse is inferred. |
| `P4 COLLISION` | **PARTIAL** | `A_FU^mc` explicitly requires both image holes fresh. Correct-role old occupancy remains covered by S53; wrong-role occupancy and missing/occupied mirror holes remain physical exits. S54 still proves no separation from fixed `T_0`. |
| `P5 SHADOW-F-TERMINAL` | **PROVEN on CAD+canonical LOCK and admitted augmented reserve traces; FORCED FAILURE on fast one-for-one histories; OPEN globally** | S59's unique-hole append is physical and co-terminal. S61 forces Definition 55.1 outcome 4 at age six for the fast class. Simultaneous legality and terminal maintenance remain separate duties elsewhere; S20 binds. |
| `P5R REAL-S-TERMINAL-REFLECTION` | **PROVEN in inherited guarded classes and `A_FU^mc`; OPEN globally** | S63 supplies an actual associated `Shat` six. S62 proves that without such supply the duty fails--P5R for debt-meeting windows and the separately restored common-only-win transfer otherwise. S14 and S25 remain mandatory. |
| `P6 CAUSALITY` | **PROVEN for every new local selector; OPEN globally** | `Pi_min` is prefix-based; catch and quiet tests use only the reached prescription; S62's holes are selected at the current real turn; S63's second image is fixed from the observed first-unsafe window. No future F cell is announced to S. |

## 70. Hostile-review attack surface and regression matrix

### 70.1 Load-bearing limitations

1. **Scalar feasibility is not a point representation.** S55 uses the
   many-to-one window portfolio exactly as defined. It merges no stones,
   recolors nothing, and proves no spatial inverse.
2. **State, canonical lock, and augmented lock are distinct.** CAD certifies
   readiness. Definition 55.3's canonical `F-LOCK` still concerns a service
   coordinate fixed independently of the terminal prescription. S59 proves
   `F-LOCK^+` only for its different explicit sequential handler.
3. **The policy is fixed before play.** Least orders and finite-prefix
   special cases totalize `Pi_min` and the reserve handler. `None` remains a
   strict-class exit, never a valid portfolio.
4. **Dynamic quietness is load-bearing.** Each reached prescription must miss
   every deficit-two window at that microstep. A condition checked only at
   pair start is too weak because the first append can create a new near
   window.
5. **The readiness cliff is first-microstep exact.** S58 proves failure before
   the mandatory second query. It does not claim that a one-hole shadow window
   survives an intervening S/`Shat` pair after a second event.
6. **An extra real append is not a repair inside S40.** It may be legal on the
   real engine, but before the paired shadow event it loses common phase and
   the one-event-per-microstep P3 semantics.
7. **Reserve rolling admission is conditional.** S59 assumes the following S
   pair is first-safe, certified, nonterminal, `tau_E<=1`, and avoids `W_*`.
   No theorem forces arbitrary S play to satisfy those clauses.
8. **A sound real win ends service duties.** S56 separates nonterminal service
   compatibility from the fallback of filling a real reserve. It never
   continues the terminal real board.
9. **Positive witnesses are diagnostic.** The S57/S59 trace uses a total legal
   extension of `sigma_star`; S63 uses the folded `C_shield` trace. Neither
   strategy is claimed winning against every `Shat` counterplay.
10. **Fast means a uniform counterplay bound.** `d_sigma(h)<=8` quantifies over
    every complete/maximal legal `Shat` counterplay from that checkpoint, not
    merely the S50 branch. S24 supplies finiteness but no value eight.
11. **The fast class may be empty.** S61 is a valid conditional strategy-own
    forcing theorem, not an existence theorem for an alleged winner.
12. **S50 remains one guarded cycle.** Its first-safety, certificate, and
    `tau_E<=2` conclusions reach local placement six only. S60, not an
    unstated iteration of S50, makes that prefix decisive for fast winners.
13. **S51 stays a non-disjoint stop cover.** Membership failure, real sound
    stop, and misaligned shadow terminal may overlap other diagnostic facts.
    S51 is not a controller.
14. **S51.1 is never crossed.** The hypothetical local placements seven and
    eight in S60 exist only under the alleged-winning premise. If a `Shat`
    append wins, that is the contradiction stop and no later state is used.
15. **The deadline theorem is a cover, not a constructive shadow win.** S62
    says `Shat` wins or the proposed handler fails terminal transfer. Only
    S63 supplies the terminal coordinate positively.
16. **Mirror-clean alleged-winning reachability is contradictory.** Its
    physical nonempty cylinder is diagnostic. The theorem's value is the
    explicit causal stop, not a continuing alleged-winning subgame.
17. **P5R and common-only wins are not synonyms.** S62 states both duties.
    Every ledger retains the restored common-only physical-transfer caveat
    and simultaneous legality plus P2/P3/P5/P5R maintenance.
18. **The high-transversal corollary needs a common nonterminal handoff.** It
    does not cover a phase-lagged handler, an earlier F terminal, or an
    already-present correct-role certificate. S45/S30 reachability is still
    absent.
19. **Second placements are sequential.** Portfolios, deficits, urgent sets,
    legal stores, certificates, and terminal tests are recomputed after each
    pair of actual nonwinning appends.
20. **Every physical stone persists.** Prepayment, reserve cells, images,
    fillers, old certificates, and proxies retain occupancy, support,
    blocking, and terminal effects. Rebinding changes no board.
21. **Local branches remain globally conjunctive.** A, B, C, service, P3,
    P5/P5R, collision, persistence, causality, and cadence obligations are
    never traded away by choosing a convenient local outcome.
22. **No outcome inflation.** Conditional maintenance plus conditional forced
    misalignment does not decide the slow class or select D2's determinacy
    alternative. `NL_F` is **OPEN**.

### 70.2 Binding regression matrix

| Regression | Round-8 treatment | Remaining boundary |
|---|---|---|
| S12 preannounced real-F coordinate | S57/S59 select prepayment, catch, and reserve cells only after the relevant S turn and reached `sigma` prescription; `W_*` constrains class membership but is not announced to real S as future play | Every outer backing/recode plan still owes S12; universal reserve avoidance is not inferred |
| S13 fixed-isometry FIFO frontier | F-role policies are temporal event pairings, not inverse/FIFO maps; S63 is a two-step terminal image completion | Every fixed-isometry one-stone FIFO proposal satisfying S13's premises remains excluded; S54 still gives no separation |
| S14 literal one-cell terminal lag | S62 exposes the exact deadline failure; S63 supplies a same-turn physical terminal only in its mirror-clean class | Unguarded literal lag and non-mirror first-unsafe continuations remain excluded/open |
| S18 proxy-supported reverse illegality | No new F-role inverse is requested; the S59 witness retains S41's legal `Fhat@(-8,0)` paired with a different real cell | Every spatial inverse carrier still owes reverse legality |
| S20 proxy-fabricated `Fhat` win | S59's augmented unique-hole lock physically transfers the S41 terminal; S61 identifies the fast count regime where no real six is possible | Universal canonical LOCK and slow-strategy terminal fidelity remain open |
| S25 older-surplus real-S win | S62 retains the terminal memory and distinguishes debt-meeting P5R from common-only transfer; S63 allows unrelated old debt but supplies an actual terminal | Every other lag/recode branch still owes shielding, certification, blocking, or physical reflection |
| S30 exact `tau_E=5` fork | S62.2 proves a conditional common-handoff stop cover after any missed two-stone service | Candidate-own reachability, positive reconciliation, phase lag, and service beyond two remain open |
| S31 six-blocker permanent fence | No installation theorem is claimed | Six-cell availability, interruption, S occupation, reconciliation, and P3 compatibility remain binding |

## 71. Objective dispositions and sharp resume point

### 71.1 Maintenance horn

**POSITIVE ON A NONEMPTY NAMED CLASS; NEGATIVE AT AN EXACT BOUNDARY.** S55
reduces admission to a scalar test. S57 gives a legal dynamically quiet
one-debt ingress, and S59 maintains `CAD+F-LOCK^+` through every admitted
reserve-one concatenation up to any finite S24 prefix, with a complete
co-terminal witness. S58 proves that when a first prescription ages a
`(mu_H,mu_R)=(2,3)` state, no canonical or augmented one-for-one coordinate
can restore readiness before the second query. Universal alleged-winner
membership and canonical `F-LOCK` remain **OPEN**.

### 71.2 Forcing horn

**POSITIVE FOR FAST WINNERS; OPEN FOR SLOW WINNERS.** S60 proves the exact
post-S15 gap `d=6` or `d>=9`. S61 therefore forces every member of
`FAST_8^{S15}` and, at root level, every root-fast alleged winner with
`B_sigma<14`--onto S49's sixth-versus-five misalignment on its own S50
history. Existence of a fast alleged winner is **OPEN**, and generic S24 does
not bound a slow winner sharply enough.

### 71.3 Ejection outcome

**MIRROR-CLEAN FIRST-UNSAFE IS COVERED BY A PHYSICAL STOP.** S63 supplies the
two actual `Shat` appends and stops at the win. S62 proves that without a
terminal supply no common-phase one-append A/B handler can survive the same
deadline; S62.2 carries that boundary to an actual nonwinning `tau_E>2`
handoff. Missing/blocked images, wrong-role or unsupported certificates,
phase lag, and positive high-transversal reconciliation remain **OPEN**.

### 71.4 Most valuable theorem and next question

The most valuable new theorem is S61: it converts S24's qualitative finite
horizon into an exact, strategy-own sixth-stone forcing result for a named
uniform fast class without crossing S51.1. The sharpest next question is:

> For every alleged-winning `sigma` with `d_sigma(h)>=9`, can S50 be extended
> by one further genuine rolling cycle so that, on `sigma`'s own history,
> either a physical stop occurs, a first-event `(2,3)` aging cliff is forced,
> or the next checkpoint is admitted to `R_1` with its reserve window
> protected against the actual S pair?

A positive answer would join the slow forcing route to the maintenance
controller. A negative answer should expose the next exact obligation--most
likely reserve-window avoidance versus first-unsafe/high-transversal
ejection--rather than another unstructured membership exit.

## 72. Provenance

### 72.1 Repository and artifact identity

- User-declared input commit: `175ca45e`, resolving to
  `175ca45e3772659f1026ff8116268f78e3b18a06`.
- User-declared branch: `hunt/gap-raw`.
- Observed branch during authoring: `hunt/gap-raw`.
- Observed worktree `HEAD` during authoring:
  `c019400ad14e06fa6f600c5462113a74795e3270`.
- `175ca45e` is an ancestor of the observed `HEAD`. The required strategy-
  stealing corpus and six Rust rule sources are unchanged between them. The
  intervening paths are coordinator prompt/raw-front artifacts and were not
  opened or used as evidence.
- Deliverable: `STRATEGY_STEALING_ROUND8.md` in the shared worktree.
- Commit made by this pass: **none**, as required.
- Landed artifact commit/hash: **not yet known** because this pass is forbidden
  to commit. The byte hash of the final working-tree artifact is reported in
  the handoff rather than self-embedded (which would change that hash).

The observed-HEAD discrepancy is recorded rather than silently rewriting the
requested proof baseline. It does not change the proof inputs. Unrelated
pre-existing or externally created untracked worktree entries were left
untouched.

### 72.2 Required corpus read first, in order and in full

The authoring pass read the prescribed files before developing the round-8
claims:

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md` and
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md` and
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including folded section 35, and
   `STRATEGY_STEALING_REVIEW_ROUND4.md`;
5. `STRATEGY_STEALING_ROUND5.md`, including folded section 44, and
   `STRATEGY_STEALING_REVIEW_ROUND5.md`;
6. `STRATEGY_STEALING_ROUND6.md`, including folded section 53, and
   `STRATEGY_STEALING_REVIEW_ROUND6.md`; and
7. `STRATEGY_STEALING_ROUND7.md`, including binding section 63, and
   `STRATEGY_STEALING_REVIEW_ROUND7.md`, whose restored twelve-obstacle list
   was treated as authoritative.

No `GAP_RAW_*` file was read, quoted, or used as mathematical evidence.

### 72.3 Rule sources read in full

The following production sources were read in full:

- `packages/hexo_engine/rust/src/coord.rs`;
- `packages/hexo_engine/rust/src/legal.rs`;
- `packages/hexo_engine/rust/src/rules.rs`;
- `packages/hexo_engine/rust/src/board.rs`;
- `packages/hexo_engine/rust/src/state.rs`; and
- `packages/hexo_engine/rust/src/tactics.rs`.

They were used only for the rule facts in section 64.2. No other source was
treated as engine-law authority.

### 72.4 Hand-proof and mutation boundary

All new results S55--S63 are hand proofs. No Cargo command, Rust build, Lean
command, harness, executable game search, solver, or proof-search program was
run. Read-only text and Git-metadata inspection were used to audit the corpus
and baseline. File creation and incremental revision used patch operations.
No existing proof/source file was modified, no artifact was
deleted, and no commit was created.

## 73. Errata and corrections folded from the round-8 hostile review

`STRATEGY_STEALING_REVIEW_ROUND8.md` (ultra, reviewed artifact `e93d9d74`,
SHA-256 `7910d291...ffcc7a`) returned overall **REFUTED** — but with a
precise split: **the local mathematical core S55-S63 SURVIVES at its
conditional scopes** (§65 CONFIRMED-WITH-QUALIFICATION, §66
CONFIRMED-WITH-MINOR-REPAIRS, §67 CONFIRMED-AT-SCOPES); what is REFUTED is
the §68 SYNTHESIS — its claim that one missing fast-or-reserve quantifier
is exactly why `NL_F` remains open. Folds:

1. **(Finding 1, REFUTED — folded)** §68 rewritten as a PARTIAL INTERFACE
   MAP: a universal fast-or-reserve theorem would connect two local
   interface results but would not prove `NL_F`; the unresolved region
   contains at least the fast S49 OUTCOME branch (S61 defeats the tested
   one-for-one carrier, not `sigma`), slow winners, the quiet/`R_1`
   admission complement, and the unresolved ejection/outer-carrier
   classes. §69's fidelity row relabeled "FAST ONE-FOR-ONE TERMINAL
   TRANSFER FORCED TO FAIL; OUTCOME OPEN."
2. **(Finding 2, MINOR)** The §68 decision table now carries its source
   hypotheses: the S57 row adds the common-live `FirstStone` + `tau_E<=1`
   conditions; the `(2,<=2)` row is split into one-event feasibility vs
   rolling maintenance conditional on every `R_1` admission clause.
3. **(Finding 4, MINOR)** S57's `tau_E=0` branch gains the fixed
   least-legal-filler clause with sound-stop closure.
4. **(Finding 5, MINOR)** S58's quantifier qualified to "any query-first,
   common-phase, one-event-per-microstep one-for-one selector" —
   asynchronous/prepayment architectures leaving the S40 carrier are
   outside the theorem.
5. **(Finding 6, MINOR)** `R_1`'s definition states the
   reserve-replaces-canonical-service semantics and §53 atomic closure
   explicitly. Every summary keeps "augmented `F-LOCK^+` only" — canonical
   `F-LOCK` is NOT proved.
6. **(Finding 7, MINOR)** S60's placement-7/8 step now uses the direct
   alleged-winningness contradiction (winning append + off-path
   totalization) instead of citing S51.1 beyond its common-live coupled
   premise.
7. **(Finding 8, MINOR)** S61 gains the complete/maximal-extension
   sentence before applying `d_sigma`, the "item 3, possibly alongside
   another cover item" wording, and the explicit root bound
   `d_sigma(h) <= B_sigma - 5 = 6`.
8. **(Finding 12, MINOR)** Landing identity `e93d9d74` recorded here (the
   audited artifact body is not rewritten to self-embed its own hash).

**Review confirmations of record.** S55/S56 censuses exact (Finding 3);
S59's induction preserves both CAD and augmented readiness with the full
witness verified clause-by-clause (Finding 6); S60's cadence table and
depth gap exact (Finding 7); S61 strategy-own and count-exact (Finding 8);
S62/S62.2 exact negative covers at named scopes (Finding 9); S63 a genuine
physical stop obeying the terminal grammar (Finding 10); the restored
carry-forward caveat ledgers are COMPLETE this round (Finding 11 — the
recurring omission class is fixed). The review's twelve-item obstacle list
supersedes §69 as the authoritative open state; its item 12 (fast-class
nonemptiness, slow-winner controller, quiet-membership preservation,
outcome-level argument after S49) is the sharpest front.
