# Strategy stealing in engine Hexo, round 6: the P3 event carrier

**Worktree:** `hunt/gap-raw` at input HEAD `3000a117`  
**Date:** 2026-07-18  
**P3-carrier verdict:** **POSITIVE-AT-SCOPE** for the terminal-aligned event
class defined in section 46; the universal alleged-winning-`sigma` carrier
remains **OPEN**.  
**Global target:** `NL_F` remains **OPEN**.

This round isolates the sharp P3 point. A shadow prescription need not be
sent through an inverse coordinate map. It can instead be paired, at the same
F-role microstep, with the canonical real service placement selected after
the preceding S turn. This event pairing is genuine and causal: `sigma`'s
coordinate is appended legally on the physical shadow board, the service
coordinate is appended legally on the physical real board, and the second
prescription is queried only after both first placements are nonterminal.
Consequently the reverse-frontier gadget S18 does not obstruct the
nonterminal event carrier, and the service coordinate is never announced
across an intervening S turn as in S12.

Event pairing alone is not terminal meaning. The carrier succeeds only when
every terminal shadow-`Fhat` event has a same-step physical real-F terminal
certificate. Section 46 proves the conditional carrier theorem, gives a
nonempty complete trace whose final paired first placements win on both
boards, and gives a candidate-own-history negative control in which a legal
`sigma` completes the S20 pattern while the canonical real service leaves F
with only five stones. The negative control refutes the named terminal-blind
selector class when promised for every legal `sigma`; it does not force that
prescription from an arbitrary alleged-winning `sigma` or refute other
selectors.

For branch (A), the support cut is exactly saturated by a budget of two
episodes per S pair. It forces one episode at each nonwinning S coordinate,
but after the second coordinate the engine passes to F and the counter resets
before S can choose again. Thus the reset genuinely escapes S32's lifetime
counting argument; no successful per-pair carrier is constructed. For branch
(C), a transversal theorem shows that a common-phase physical deficit
certificate cannot replace the missing blockers at a `tau_E>2` service
state. In particular it cannot carry the abstract S30 fork across a
nonwinning two-stone F turn without physical reconciliation, an earlier sound
stop, or departure from Definition 39.1. S30 is still not proved reachable on
an arbitrary candidate's winning-`sigma` history.

All results are hand proofs. No Cargo command, Lean build, harness,
executable search, or proof search was run. This authoring pass creates no
commit.

## 45. Statement boundary and binding inherited state

### 45.1 Target, roles, and status discipline

Let `F=Player0` be the compulsory real opener and `S=Player1` the real second
player. In the role-swapped shadow, `Shat` represents real S and is the
opener, while `Fhat` represents real F and follows a fixed alleged-winning
second-player strategy `sigma`. The target remains

`NL_F : exists pure sigma_F, for every pure sigma_S, S never wins`.

As in the earlier rounds, `S never wins` permits a finite F win or an infinite
nonterminal history with neither owner completing six. Round-2 Theorem D2 is
inherited **[PROVEN from the CITED Gale--Stewart open-determinacy theorem]**
on the declared unbounded-board macro-game:

`NL_F <=> S has no winning strategy`.                         (D2)

This document does not refute every alleged-winning `sigma`, so D2 does not
yield `NL_F`. Every named or load-bearing claim below is marked **PROVEN**,
**SKETCH**, **CONJECTURE**, **OPEN**, or **CITED**. Definitions are
stipulative. `POSITIVE-AT-SCOPE`, `NEGATIVE-AT-SCOPE`, and `PARTIAL` are used
only as requested objective-disposition labels, never as proof statuses.
There are no machine-verified claims.

### 45.2 Production rules used [PROVEN]

On axial coordinates, the executable distance is

`d((q,r),(q',r'))=max(|q-q'|,|r-r'|,|(q-q')+(r-r')|)`.

The only opening is `F@(0,0)`. Along nonterminal forward play the owner
cadence is

`F ; S,S ; F,F ; S,S ; F,F ; ...`.

A normal placement is legal exactly when its coordinate is physically empty
and belongs to the color-blind radius-eight legal store. A nonwinning first
placement is inserted before the second placement is validated. A six in a
Q, R, or QR length-six window is checked after every single append and before
phase advancement; a winning first placement suppresses the second. A
terminal state has no legal continuation.

These facts are implemented in
`packages/hexo_engine/rust/src/coord.rs:1-4,9-20,76-95`,
`legal.rs:17-18,123-145`, `rules.rs:11-44`, `board.rs:83-105`,
`state.rs:149-160,203-252,289-357`, and
`tactics.rs:13-17,21-75,205-208,451-485`. Forward placement inserts into
physical occupancy and history; the public apply/undo API is an analysis/MCTS
path, not a legal-history operation. Thus every old proxy, filler, service
stone, and queued stone remains physically present. The executable carrier is
`i16`; the general theorems retain the inherited `Z^2` idealization, and each
explicit witness below is separately bounded inside the safe carrier region.

### 45.3 Binding inherited theorems and errata

The following scopes are mandatory.

- S12 excludes a still-empty real-F first coordinate fixed before an
  intervening S turn under its subset/support premises.
- S13 excludes its fixed-isometry one-cell FIFO schedule; S14 excludes an
  unguarded literal one-S-stone terminal lag.
- S18 refutes unconditional inverse-coordinate shadow-to-real legality.
- S20 refutes terminal transfer through a proxy-assisted shadow-`Fhat` win.
- S25 makes older-real-surplus terminal memory physical, not a label issue.
- S30 gives a legal abstract labeled fork with three disjoint urgent hole
  pairs, hence inherited `tau_E>=3`, but no candidate-own-history shadow
  certificate. Section 48 sharpens its exact transversal separately.
- S31 gives the exact unconstrained six-blocker cost of a permanent fence.
- S32 refutes only lifetime `K=2` in the total-exact isometric branch; two
  episodes per S pair remain open.
- S33/S34 prove the rolling guarded queue and exact real-board service on
  finite `A_FS2` segments, including the transient two-debt microstep.
  Every physical branch-(B) `Shat` append, reconciliation or filler, stops
  the module if it wins; it contradicts `sigma` only under a genuine outer
  strategy-domain carrier.
- S36--S38 define and transfer physical window-deficit certificates; S39.1
  prevents that natural common-phase certificate from carrying shield-unsafe
  debt on a winning-`sigma` history.

Round-2, round-3, round-4 section 35, and round-5 section 44 errata are folded
into every use. In particular, the physical histories are append-only;
candidate grammar is distinct from extensional success; and an abstract
legal gadget is not a negative strategy-specific obstruction.

### 45.4 Correct quantifiers for branches (A), (B), and (C)

For every alleged-winning `sigma`, every strategy-generated genuine live
prefix, and each observed legal real-S single placement, a global candidate
may choose exactly one local response type:

- **(A)** a zero-lag repair completed before the next engine placement;
- **(B)** an explicit lag/queue satisfying its phase-sensitive P5R guard; or
- **(C)** an actual same-step physical shadow terminal certificate.

These are alternatives only for the current real-S placement. In **every**
branch the candidate must also provide recurring P3 transfer, real-board
service or physical reconciliation, P5 and P5R terminal fidelity, causal
selection, strategy-domain legality, and accounting for every persistent
physical stone. It must pass S12, S13, S14, S18, S20, S25, S30, and S31 at
their inherited scopes. A negative gadget counts against a strategy-specific
candidate only when the adversarial choices are selected from that
candidate's own legal `sigma`-consistent coupled history.

## 46. The P3 carrier

### 46.1 Canonical ordered service and event pairing

At a handled F `FirstStone` checkpoint of an `A_FS2` segment, let `K_E` be
Round 5's first minimum transversal of the urgent E-live hole family. When
`|K_E|<=2`, order its members by the fixed service enumeration and then,
after each nonwinning placement, append the least fresh legal padding
coordinate until two coordinates have been selected. Write the resulting
ordered real service as

`svc_E(R)=(k_1,k_2)`,                                      (46.1)

with `k_2` understood to be suppressed if `F@k_1` wins. S29/S33 prove that
every required transversal member is already real-legal and that the padding
rule is sequentially legal. The selector is evaluated only after the
preceding S pair, and no S action intervenes before F uses it.

**Definition 46.1 (P3 event carrier).** Fix one genuine shadow history and a
pure strategy `sigma`. At each covered common F-role `FirstStone` checkpoint,
the carrier:

1. computes (46.1) from the observed real prefix;
2. queries `sigma` for its legal shadow first prescription `z_1`;
3. pairs the two physical events `(Fhat@z_1, F@k_1)` and appends each on its
   own board;
4. stops soundly on a real F win; otherwise, if `Fhat@z_1` wins, accepts the
   event only when `F@k_1` is also a physical real-F win; and
5. if both first placements are nonterminal, queries the now-reached
   `SecondStone` prescription `z_2`, pairs it with `k_2`, appends both, and
   applies the same terminal rule.

The temporal pairing relation is

`Q_F={(z_i,k_i): paired F-role microsteps already appended}`.              (46.2)

At each pre-event state define the physical immediate-win sets

`D_H^F={z in L(O_H): some window V has V\{z} subseteq X_Fhat}`,

`D_R^F={k in L(O_R): some window W has W\{k} subseteq X_F}`.                (46.3)

The exact terminal bridge is

`z_i in D_H^F  =>  k_i in D_R^F`.                         (46.4)

The sets and implication are recomputed after a nonwinning first event.
Equation (46.4), not the mere existence of `(z_i,k_i)`, is the physical
meaning of event-terminal alignment.

It is not a coordinate map. In particular it makes no assertion that
`k_i=T^{-1}(z_i)`, that the two coordinates have the same support, or that
old F-role stones are pointwise represented. After each nonterminal paired
append, both physical legal stores, all physical window masks, the rolling
debt set, the future certificate collision `T(E_S) intersect O_H`, and the
next urgent family are recomputed from the append-only boards. A future
rolling step is covered only if the resulting prefix again satisfies the
`A_FS2` freshness, first-safety, nonterminal-second, and `tau_E<=2`
conditions.

Call a finite or terminal trace **event-terminal aligned** when every covered
shadow-`Fhat` terminal append is paired with a real-F terminal append on that
same microstep. Let `A_FS2^ET(sigma)` be the `A_FS2` trace segments whose
F-role steps are generated by Definition 46.1 and are event-terminal aligned.
This definition does not treat either a terminal shadow win or a failed
old-debt certificate as a successful live continuation: a terminal event
must use its stated physical stop rule, while a certificate failure removes
the trace from the class and remains an outer coverage obligation.

### 46.2 Positive theorem at the named scope

**Theorem S40 (causal service-event carrier) [PROVEN at `A_FS2^ET` scope].**
For every pure shadow strategy `sigma` and every finite trace in
`A_FS2^ET(sigma)`, Definition 46.1 realizes each reached sequential
`Fhat` prescription as one legal real-F service event and preserves the
common F-role `FirstStone`/`SecondStone` cadence until a sound physical stop.
More specifically:

1. every shadow append is in one genuine legal history and every `Fhat`
   append is the prescription of `sigma` at that exact history;
2. every paired real append is the corresponding member of the canonical
   ordered service (46.1) and is legal at its actual real phase;
3. S18-type proxy-only support of `z_i` cannot make `k_i` illegal, because no
   inverse-coordinate legality claim is made;
4. every reached shadow-`Fhat` terminal placement, including a
   second-placement S20 pattern, has a same-step physical real-F terminal
   certificate; and
5. no real-F coordinate is fixed across an intervening S turn.

If `sigma` is alleged winning and a complete outer coupling keeps every
opponent continuation inside `A_FS2^ET(sigma)` until `sigma` stops, then the
real play reaches a finite F win no later than that stop. This last sentence
is conditional on the outer coverage premise; S40 does not prove that every
legal real-S continuation remains in the class.

*Proof.* At a covered F `FirstStone` checkpoint, the preceding S pair and all
of its physical queue updates are already known. Hence `K_E`, its order, and
the first padding choice are functions of the observed prefix. S29/S33 give
real legality of every transversal member; the definition of padding gives
legality of padding. Independently, `sigma` is queried on the genuine shadow
prefix and returns a shadow-legal `z_1`. Appending `k_1` on the real board and
`z_1` on the shadow board therefore realizes one legal event on each board.

If either first append wins for real F, the real objective has reached a
sound stop. If `z_1` wins, event-terminal alignment supplies the real-F win on
that same paired step. Otherwise both engines advance the F role from
`FirstStone` to `SecondStone`. Only now is `sigma` queried for `z_2`; the
canonical service selector chooses or validates `k_2` in the real post-`k_1`
state. The same argument proves legality and terminal transfer for the second
events. If both are nonwinning, both engines pass to the S role at
`FirstStone`. The `A_FS2` clauses then govern the next physical queue cycle.

S18 concerns the failed implication
`z in L(O_H) => T^{-1}(z) in L(O_R)`. The carrier never uses that implication:
`z_i` remains on the shadow board and the separately supported `k_i` remains
on the real board. Cross-board equality, support, and vacancy are therefore
irrelevant; same-board vacancy and support were just proved. All resulting
occupancy, support, window, and future-certificate effects persist and are
recomputed as required by Definition 46.1.

Finally, both `K_E` and padding are selected after S has completed its turn,
and F acts before S acts again. S12's intervening-S premise is absent. If a
fixed alleged-winning `sigma` stayed nonterminal forever on the genuine
shadow counterplay, it would contradict its winning property (equivalently,
S24 supplies a finite uniform horizon from each fixed checkpoint). Thus under
the displayed complete-coverage premise `sigma` eventually gives a terminal
`Fhat` event, and event-terminal alignment gives a real F win no later. QED.

S40 discharges P3 legality, cadence, and causality only on its named trace
class. It explicitly delegates coverage of first-unsafe, certificate-failing,
and `tau_E>2` S continuations; maintenance of the rolling S-role certificate;
and construction of the physical F-terminal alignment condition for an
arbitrary alleged-winning `sigma`.

### 46.3 A complete nonempty terminal-aligned trace

**Lemma S41 (S18-stressed, proxy-terminal service trace) [PROVEN].** Fix

`T(q,r)=(q+2,r)`

and one coordinate enumeration beginning

`(1,0) < (2,0) < (0,1) < (1,5) < (3,0) < (4,0) < (5,0) < ...`.

Among equal-size transversals put `{(0,5)}` before `{(0,6)}`. Let
`sigma_star` have the following on-path `Fhat` prescriptions and extend it at
off-path nodes by the least legal coordinate.

| stage | real append | shadow append |
|---|---|---|
| S15 prefix | `F@(0,0); S@(0,1),S@(0,2)` | `Shat@(0,0); Fhat@(1,0),Fhat@(2,0); Shat@(2,1),Shat@(2,2)` |
| seed service | `F@(1,0),F@(2,0)` | `Fhat@(3,0),Fhat@(4,0)` |
| rolling S pair 1 | `S@(0,3),S@(0,4)` | `Shat@(0,1),Shat@(2,3)` |
| service 1 | `F@(0,5),F@(1,5)` | `Fhat@(2,5),Fhat@(3,5)` |
| rolling S pair 2 | `S@(1,4),S@(2,4)` | `Shat@(2,4),Shat@(3,4)` |
| service 2 | `F@(3,0),F@(4,0)` | `Fhat@(-8,0),Fhat@(5,0)` |
| rolling S pair 3 | `S@(8,0),S@(8,1)` | `Shat@(4,4),Shat@(10,0)` |
| terminal service | `F@(5,0)` | `Fhat@(6,0)` |

This is a complete finite member of `A_FS2^ET(sigma_star)`. Its second
service directly realizes an S18-type prescription whose inverse is illegal,
and its final paired first placements simultaneously complete a real F window
and a proxy-assisted shadow-`Fhat` window.

*Proof: legality and queue service.* The S15 prefix is exact:
`Fhat@(1,0)` is the persistent proxy and `Fhat@(2,0)=T(0,0)` represents the
real opening. The seed service is canonical padding from `K_E=empty`; the
first two enumerated coordinates are real-legal and its shadow prescriptions
are adjacent legal cells.

During rolling pair 1, real `(0,3)` is first-safe and shadow `(0,1)` is the
first fresh legal filler. Real `(0,4)` is nonwinning and shadow
`T(0,3)=(2,3)` physically reconciles the old debt. The sole urgent unblocked
window for debt `(0,4)` is

`{(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)}`,

whose holes are `(0,5),(0,6)`. Thus `K_E={(0,5)}` and the first eligible
padding coordinate is `(1,5)`. Shadow `(2,5),(3,5)` is a legal nonwinning
pair.

Rolling pair 2 physically reconciles `(0,4)` at `(2,4)`, then reconciles its
first coordinate `(1,4)` at `(3,4)`, leaving debt `(2,4)`. Every window
through the first new debt has post-deficit at least four, the second real
placement is nonwinning, and every window through `(2,4)` contains at most
three real S stones. Hence `tau_E=0`; after skipping occupied enumeration
cells, canonical service is real `(3,0),(4,0)`. Both shadow prescriptions are
legal and nonwinning.

Rolling pair 3 reconciles `(2,4)` at `(4,4)`, then reconciles `(8,0)` at
`(10,0)`, leaving `(8,1)`. Real `(8,0)` is first-safe, supported by
`F@(4,0)` at distance four; `(8,1)` is adjacent and nonwinning. Every window
through the final debt contains at most two real S stones, so `tau_E=0` and
the next canonical padding coordinate is `(5,0)`.

Every support check other than `(-8,0)` is at distance at most five. The
exceptional `(-8,0)` is supported by the opener at exactly eight; `(10,0)` is
supported color-blind by `Fhat@(5,0)` at distance five; every other
reconciliation cell is within distance two of a prior `Shat` cell. Every
coordinate is fresh at its append.

*Proof: direct S18 stress.* At service 2, `sigma_star` prescribes
`z_1=(-8,0)`. It is shadow-legal solely through persistent `Shat@(0,0)`:

`d((-8,0),(0,0))=8`,

while every other then-occupied shadow cell is farther away. Its rolling-map
inverse is `T^{-1}(-8,0)=(-10,0)`. Every then-occupied real cell has
nonnegative q-coordinate, and the closest is `(0,0)` at distance ten, so the
inverse is real-illegal. The event carrier instead pairs `z_1` with canonical
real service `k_1=(3,0)`, adjacent to `F@(2,0)`. At the final event inverse
copying also fails by occupancy:

`T^{-1}(6,0)=(4,0)`,

already real-F occupied; the carrier uses fresh service cell `(5,0)`.

*Proof: terminal bridge and absence of earlier wins.* Immediately before the
last event, real F has the five-cell q-axis run `(0,0),...,(4,0)`, and shadow
`Fhat` has `(1,0),...,(5,0)`. Their other stones form runs of length at most
two. Real S has a longest run of four on `q=0`; shadow `Shat` has a longest
run of four on `q=2`. Hence both boards are nonterminal, and every earlier
prefix, being a subset, was nonterminal. The paired cells complete

`W_real={(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)}`

and

`V_shadow={(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)}`.

The shadow window contains the original `Fhat` proxy `(1,0)`, so this is a
physical transfer of the proxy-assisted S20 phenomenon, not a proxy-free
special case. Both wins occur at `FirstStone` and suppress the seconds. The
largest displayed norm is ten and every update halo has norm at most
eighteen, safely inside `i16`. QED.

The on-path prescriptions extend to a total legal pure strategy by the
least-legal rule. `sigma_star` wins on this displayed counterplay but is **not
claimed winning against every shadow-opener strategy**. S41 proves physical
nonvacuity and exercises both S18 and S20; it does not establish universal
membership for alleged-winning strategies.

### 46.4 Negative control: the named terminal-blind pairing dies on its trace

**Definition 46.2 (`C_evt^blind[prec_dagger]`).** Fix the named canonical
selector `prec_dagger` whose reached real padding order is
`(1,0),(2,0),(3,0),(4,0),...` and whose first reached shadow filler is
`(0,1)`. A terminal-blind event carrier in this class has the same genuine
histories and causal timing as Definition 46.1 and uses this selector, but
promises full P3/P5 carrier correctness for every legal pure `sigma`, every
legal S15 synchronization for it, and every rolling continuation generated by
the fixed queue and selector that satisfies the `A_FS2` admission clauses,
through the first physical terminal event. It attempts to justify that promise
using only the temporal pairs (46.2): it neither requires nor constructs a
physical real-F certificate when a paired shadow-`Fhat` placement wins. It may
not voluntarily prune the covered cylinder, truncate a live continuation, add
an unmatched real placement, change the selected canonical service, or
continue the terminal shadow history.

**Theorem S42 (own-history terminal obstruction to
`C_evt^blind[prec_dagger]`) [PROVEN].** No carrier in the named class satisfies
its universal promise. There is one legal pure strategy `sigma_dagger`, one
S15 synchronization, and one rolling
first-safe/two-serviceable continuation on the carrier's own genuine
`sigma_dagger`-consistent history such that `sigma_dagger`'s second service
pair wins on its second placement while the canonical real service leaves F
with only five stones. No legal service-compatible real-F sequence allowed by
Definition 46.2 transfers that terminal event.

*Proof.* Use `T(q,r)=(q+2,r)`, the same real service padding order
`(1,0),(2,0),(3,0),(4,0),...`, and the first shadow filler `(0,1)`. Play

```text
real:
F@(0,0);
S@(0,2),S@(2,1);
F@(1,0),F@(2,0);
S@(3,3),S@(5,2);
F@(3,0),F@(4,0).

shadow:
Shat@(0,0);
Fhat@(1,0),Fhat@(2,0);
Shat@(2,2),Shat@(4,1);
Fhat@(3,0),Fhat@(4,0);
Shat@(0,1),Shat@(5,3);
Fhat@(5,0),Fhat@(6,0).
```

The first five shadow placements form a legal S15 prefix with proxy `(1,0)`
and represented real opening `(2,0)`. Extend the displayed `Fhat` choices to
a total legal `sigma_dagger` by least-legal off-path prescriptions. The first
canonical real service and its shadow pair are fresh, adjacent along the
q-axis, and nonwinning.

After real `S@(3,3)`, every incident axis line contains only that new S stone,
so the first coordinate is first-safe. After real `S@(5,2)`, every incident
window contains at most two S stones; the placement is nonwinning, and after
the physical filler/reconciliation update the sole debt `(5,2)` has
`delta>=4` in every E-live window. Hence `tau_E=0`, and the canonical next service is exactly
`(3,0),(4,0)`. Its two placements give real F only
`{(0,0),(1,0),(2,0),(3,0),(4,0)}`, so the real board is nonterminal. The
paired shadow placements extend `Fhat@(1,0),...,(4,0)` first to five and then
to the terminal six `(1,0),...,(6,0)`. Every move is selected on the
carrier's actual history; no abstract labeled checkpoint is imported. The
largest displayed hex norm is eight, and every radius-eight update halo has
norm at most sixteen, so this witness lies inside the safe `i16` carrier
region.

The shadow engine is terminal after `(6,0)`, while the real F turn has ended
nonterminally and no third F placement is legal in that turn. The canonical
service was fixed only after the S turn, so S12 is not involved. The failure
is solely P5 terminal meaning. QED.

S42 is the required negative attack at an exact named carrier class. Its
`sigma_dagger` is legal but not proved alleged-winning. Therefore it refutes a
carrier promised uniformly for all legal strategies and proves that terminal
alignment is load-bearing for that class; it does **not** refute a different
selector or a carrier whose domain is only the unknown class of globally
winning strategies and whose invariant somehow excludes this trace.

## 47. Branch (A): per-pair `K=2`

### 47.1 Reset grammar

**Definition 47.1 (`G_A^{2/pair}`).** Take the total-exact, owner-faithful,
translation/D6, zero-lag candidate grammar of Round-5 Definition 37.1, with
all of its genuine-history, P3, persistence, and terminal obligations. Replace
only its lifetime counter by the following counter: at each real S
`FirstStone`, set the available number of S-coordinate-reactive episodes to
two; charge at most one closed episode to the observed coordinate that
triggered it; after a nonwinning `SecondStone` passes control to F, discard
the old counter; and create a fresh counter only when S next reaches
`FirstStone`. An episode may not remain open across a coordinate. Lag,
window-only certificates, non-total maps, and unmatched physical placements
remain outside the grammar.

Let `Success_A^{2/pair}(C)` mean that the candidate covers every legal real-S
continuation until a sound physical stop and satisfies all inherited
obligations. The extensional success class is

`C_A^{K=2/pair}={C in G_A^{2/pair}: Success_A^{2/pair}(C)}`.

### 47.2 The support cut is tight at one pair

**Theorem S43 (two-cut saturation and reset escape) [PROVEN].** Fix an
alleged-winning `sigma`, one S15 synchronization, and a candidate
`C in G_A^{2/pair}`. From every candidate-own common-live S `FirstStone`
checkpoint after a successful preceding transfer, S can choose an S22 cut
coordinate and, after a successful first repair, choose a second cut
coordinate from the candidate's then-current binding. If both adaptive cuts
and their associated shadow appends are nonterminal, either a
transfer/restoration promise fails or the method forces exactly one charged
episode at each coordinate, exhausting that pair's budget of two. It cannot
force a third charged episode before the reset.

At the first tested pair after S15, both cut coordinates are necessarily
nonterminal, so the two-episode lower bound is unconditional once the
preceding `sigma` pair transfers. If both repairs succeed and the next
`sigma` pair is also nonterminal and transfers, S32's coordinate `c_3` is the
first coordinate of the **next** S pair and is charged to the fresh counter,
not to the exhausted one. Hence S32's lifetime counting proof does not refute
`C_A^{K=2/pair}`.

*Proof.* After a successfully transferred first `sigma` pair, the candidate
is at the Round-5 checkpoint

`real (F,S)=(3,2)`, `shadow (Fhat,Shat)=(4,3)`,

with S/`Shat` at `FirstStone`, a committed total exact binding, and one
physical proxy of each shadow role. S22 applied to the candidate's actual
binding supplies a real-empty legal `c_1` whose exact target is occupied.
Real S then owns only three stones, so `c_1` is nonwinning. A zero-lag total
exact candidate must fail or close its first charged episode before the next
coordinate.

After successful restoration, S/`Shat` are at `SecondStone`, the represented
and proxy parts are again nonempty, and the candidate must commit again.
S22 supplies a fresh legal `c_2`; it cannot equal occupied `c_1`. Real S then
owns only four stones, and the associated single shadow append gives `Shat`
at most five, so neither can win. The candidate must fail or close its second
episode. A successful nonwinning `SecondStone` immediately passes the engine
to F `FirstStone`. S has no third coordinate at which to select another cut
preimage, and Definition 47.1 resets only when S later regains
`FirstStone`.

Cross the next genuine `sigma` pair exactly as S32 does. Transfer failure is
already a P3 failure; a terminal sixth shadow-`Fhat` stone with only five real
F stones is already a P5 failure. In the remaining nonterminal branch, the
candidate reaches

`real (F,S)=(5,4)`, `shadow (Fhat,Shat)=(6,5)`,

at a new S `FirstStone`. The reset has occurred. S22's `c_3` is nonwinning
because it gives real S only five stones, but it consumes episode one of the
new pair. If its repair succeeds, a second cut may consume episode two at the
new `SecondStone`; that coordinate could be terminal once S has six total, in
which case the candidate owes P5R rather than a third same-pair episode. If it
is nonterminal, the pair ends and the counter resets again.

At every later common-live pair, S again selects the S22 cut at
`FirstStone` and, conditional on successful restoration, the next cut at
`SecondStone`. If either selected cut is terminal, the candidate owes P5R;
against alleged-winning `sigma`, no same-step legal `Shat` win is available on
that genuine prefix. Otherwise the two nonterminal cuts force the two charged
episodes just proved. Cadence supplies no third S coordinate before the
counter resets. QED.

### 47.3 A colored obstruction when the isometry is fixed within the pair

**Definition 47.2 (`G_A^{2/pair,static-T}`).** This is the subgrammar of
`G_A^{2/pair}` in which, after the candidate's final pre-turn commitment, the
same total isometry `T` must remain in force through both coordinates of that
S pair. The candidate may recompute the represented/proxy complement, back a
same-role proxy, and append one legal physical `Shat` filler in each episode,
but it must restore a same-phase total exact owner-faithful binding under that
same `T`. The isometry and budget may reset between S pairs.

At each such pair checkpoint, the one-stone offset per role forces the
physical complement to be exactly

`P={p_S,p_F}`,                                               (47.1)

where `p_S` is the unique `Shat` proxy and `p_F` the unique `Fhat` proxy.
This complement is forced by `A=T[O_R]`; it is not a freely movable proxy
label while `T` is fixed.

**Theorem S43.1 (colored two-proxy cut) [PROVEN].** For every
alleged-winning `sigma`, every candidate in `G_A^{2/pair,static-T}`, and every
candidate-own common-live S `FirstStone` checkpoint satisfying (47.1), real S
has a legal continuation of length at most two coordinates on which the
candidate fails before completing the pair. Hence the pair-static
per-pair-`K=2` success class is empty.

*Proof.* The genuine physical shadow support graph is connected by S21.
First suppose `p_F` has a radius-eight neighbor `a in A`. Put

`c=T^{-1}(p_F)`.

Because `p_F` is outside `A=T[O_R]`, `c` is real-empty; the inverse of `a` is
real-occupied and supports `c`, so `S@c` is legal. If `c` is nonwinning,
restoration under the fixed `T` is impossible: total owner fidelity would
require physical `T(c)=p_F` to be `Shat`, but the persistent stone is
`Fhat`. No filler or label change recolors it. If `c` wins, P5R requires an
actual legal `Shat` win on the associated step. None exists on this genuine
`sigma`-consistent prefix: any such finite winning append, together with the
reached on-path choices and least-legal off-path choices, would be a total
shadow-opener counterstrategy defeating alleged-winning `sigma`.
Thus the candidate fails on the first coordinate.

It remains to suppose that `p_F` has no neighbor in `A`. Take a shortest path
in the connected support graph from `p_F` to `A`. The proxy side has only the
two vertices in (47.1), so the path must begin

`p_F -- p_S -- a`, with `a in A`.                            (47.2)

Set `c_1=T^{-1}(p_S)`. As in S22, it is real-empty and legal through
`T^{-1}(a)`. It cannot be terminal: if it completed a real-S window, the
other five real cells would already map to five physical `Shat` stones and
`p_S` would be the sixth cell of the image window, contradicting the live
shadow premise.

The same-role occupied target forces the first episode or an immediate
failure. Assume the episode succeeds and appends its actual fresh legal
`Shat` filler `w`. With `T` fixed, the new represented set is forced to be

`A_1=T[O_R union {c_1}]=A union {p_S}`,

and the physical complement is forced to be

`P_1={p_F,w}`.                                              (47.3)

A filler-created `Shat` win would already contradict alleged-winning
`sigma`, so a successful episode reaches common-live `SecondStone`. The old
physical edge `p_S--p_F` from (47.2) persists, even though `p_S` is now
represented. Put `c_2=T^{-1}(p_F)`. Equation (47.3) makes `c_2` real-empty;
the persistent edge gives `d(c_2,c_1)<=8`; and injectivity gives
`c_2!=c_1`. Hence `S@c_2` is legal at `SecondStone` and hits the opposite-role
proxy. A terminal `c_2` fails P5R by the same no-`Shat`-win argument; a
nonterminal `c_2` cannot be represented owner-faithfully under the fixed `T`.
The second episode therefore cannot close. Every selected coordinate and the
filler `w` belong to the candidate's own history. QED.

The proof identifies the exact escape. A broader within-pair recode must
change the binding after `c_1` so that at least one endpoint of the persistent
edge changes sides: old `p_S` must again become unrepresented, or old `p_F`
must become represented while another `Fhat` stone becomes proxy. Merely
choosing a filler with different adjacency or renaming proxies cannot do so.

**Per-pair objective disposition: PARTIAL.** S43 proves that the reset genuinely
escapes S32's pigeonhole. S43.1 proves that the pair-static-isometry candidate
grammar has empty extensional success class even though its budget resets.
The full class allowing a new total binding inside each episode remains
**OPEN**; neither theorem constructs its repairs or resolves the intervening
P3/P5 forks.

## 48. Coverage outside `A_FS2`: the `tau_E>2` certificate barrier

### 48.1 A deficit certificate cannot discount a missing physical blocker

At a real F `FirstStone` checkpoint retain the inherited urgent family

`U_E={W: W intersects E_S, W intersects X_F=empty, delta_R(W)<=2}`

with physical hole sets `H_W=W\X_S` and transversal number `tau_E`. A
completed real F pair `K={k_1,k_2}` **physically hits** `W` exactly when
`K intersect H_W` is nonempty. Because `W` initially contains only real S
stones and empty holes, those holes are the only cells of `W` that F can
legally occupy.

**Theorem S44 (no common-phase certificate discount) [PROVEN].** Fix an
alleged-winning pure shadow strategy `sigma`. Suppose:

1. a genuine coupled prefix is at common F-role `FirstStone` with current
   physical debt set `E_S` and urgent family `U_E`;
2. `E_S` is held fixed through a completed nonwinning real F pair `K`, and no
   member of `E_S` is physically certified or otherwise reconciled during
   that turn;
3. the paired shadow actions form a genuine nonwinning `sigma`-consistent
   `Fhat` pair, so the successor is a common, nonterminal
   S/`Shat FirstStone` node; and
4. at that successor there exists any physical window-deficit certificate of
   Definition 39.1, with its selector allowed to be chosen or reselected from
   the full current history.

Then `K` intersects `H_W` for every `W in U_E`. Consequently `tau_E<=2` is a
necessary condition, not merely S29's sufficient service condition, for
crossing this F turn into a common-phase winning-`sigma` deficit-certified
state without reconciliation or a sound earlier stop.

*Proof.* Suppose `K` misses `H_W` for some pre-turn urgent `W`. No F stone was
in `W` before the turn, and F can enter `W` only through `H_W`; hence `W`
remains F-unblocked. The F pair changes neither `X_S` nor the fixed debt set,
so `W` remains E-live with the same

`delta_R(W)<=2`.

At the successor shadow-opener `FirstStone`, the phase deadline is `m=2`.
For any Definition 39.1 selector `nu`, S39 gives

`delta_H(nu(W))>2`

because the physical shadow prefix is genuine, nonterminal, and consistent
with alleged-winning `sigma`. The certificate inequality gives

`delta_H(nu(W))<=delta_R(W)<=2`,

a contradiction. Thus every urgent hole set must be hit. A two-stone pair is
then itself a transversal, proving `tau_E<=2`. The proof never fixes `nu`, so
dynamic reselection at the successor does not help. QED.

The theorem permits all genuine physical effects of old fillers and proxies;
they enter the shadow deficits used by S39. It excludes only a physical
reconciliation of the named real debt during the F turn. A binding change
that points to an already existing correct-role shadow stone counts as
reconciliation only if it supplies the actual physical certificate required
by the outer invariant; a label alone does not.

### 48.2 Application to the S30 three-axis fork

**Corollary S44.1 (S30 survives the natural window-certificate route)
[PROVEN at the abstract labeled-state scope].** At S30's displayed F
`FirstStone` checkpoint, the three urgent windows through `e=(0,1)` have
pairwise-disjoint hole pairs

```text
{(4,1),(5,1)}, {(0,5),(0,6)}, {(4,-3),(5,-4)}.
```

Thus every nonwinning physical F pair misses at least one window. If the debt
label `E_S={e}` is retained and not physically reconciled, no genuine
nonterminal alleged-winning-`sigma`-consistent shadow pair can carry that
state to a common `Shat FirstStone` node possessing a Definition 39.1
certificate.

Even postponing the certificate check until after S's next first placement
does not repair this fixed-debt branch **if** the handler appends one legal
nonwinning `Shat` stone and remains genuine, `sigma`-consistent, and at the
common phase. In the missed S30 window, S's offset-four hole is legal and, by
S30's cross-line check, nonwinning. Filling it leaves `delta_R=1` at common
`SecondStone`, where `m=1`; S39.1 again requires `delta_R>1`, a contradiction.
Before the offset-five terminal move, the handler must therefore physically
reconcile the debt, obtain an actual same-step physical `Shat` stop (which
would refute `sigma`), obtain an earlier sound real-F stop, or leave the
common-phase/Definition-39.1 premises.

*Proof.* Pairwise-disjointness gives `tau_E>=3`, while an F turn supplies at
most two physical cells. Apply S44. For the delayed check, use the verified
nonwinning offset-four continuation from S30 and apply S39.1 with `m=1`. QED.

**Lemma S45 (exact S30 service transversal) [PROVEN].** At S30's abstract
labeled checkpoint the full urgent family has

`tau_E=5`,

not merely the inherited lower bound three obtained from the three selected
positive-direction windows.

*Proof.* Work axis by axis through `e=(0,1)`. On the Q line `r=1`, real S
occupies q-indices `0,1,2,3`, and no real F stone lies on the line. Of the six
windows through index zero, exactly the starts `-2,-1,0` are urgent. Their
hole-index pairs are

`{-2,-1}`, `{-1,4}`, and `{4,5}`.                         (48.1)

The first and third pairs are disjoint, so at least two cells are needed;
indices `-1` and `4` hit all three, so the exact Q-axis cost is two.

On the R line `q=0`, real S occupies r-indices `1,2,3,4`. Real `F@(0,0)`
blocks every incident candidate except the window with r-start `1`. Its holes
are `(0,5),(0,6)`, so the exact R-axis cost is one.

On the QR line `q+r=1`, parameterize by q. Real S occupies q-indices
`-4,0,1,2,3`, and no real F stone lies on this line. The only urgent starts
are again `-2,-1,0`, with the same interval pattern (48.1); its exact cost is
two. The three axis lines meet only at `e`, which is S-owned and belongs to no
hole set. Hence no chosen blocker can serve two axes, and the lower bounds add
to `2+1+2=5`. Conversely, choose indices `-1,4` on each of the Q and QR lines
and either R-axis hole. Those five empty cells hit every urgent hole set,
proving the upper bound. QED.

Thus two service stones leave a deficit of three relative to the exact
transversal, although S30 needed only its simpler three-window subfamily to
refute blanket two-stone service.

**S30 objective disposition: NEGATIVE AT A CONDITIONAL MODULE SCOPE; OPEN
globally.**
S44.1 extends S39.1: the natural physical deficit certificate cannot replace
the missing service blockers. It does not prove that an arbitrary carrier ever
reaches S30's semantic label on its own alleged-winning-`sigma` history, and
it does not exclude physical reconciliation, phase-lagged event
certificates, branch (A), or a genuine branch-(C) stop. Accordingly S30
remains a mandatory stress case, not a global negative gadget.

## 49. Result ledger and objective dispositions

### 49.1 New theorem ledger

| Claim | Status | Exact scope |
|---|---|---|
| Definition 46.1 event carrier | **Definition** | Temporal pairing of sequential `sigma` prescriptions with canonical post-S real service; no F-role coordinate map |
| S40 causal service-event carrier | **PROVEN AT SCOPE** | Every finite `A_FS2^ET(sigma)` trace; genuine P3 legality/cadence, local S12 avoidance, S18 independence, and physical P5 transfer |
| S41 S18/S20 terminal trace | **PROVEN** | One complete finite physical trace with active queue/service, illegal inverse `(-10,0)`, occupied final inverse, and simultaneous proxy-assisted terminal windows; `sigma_star` legal, not claimed globally winning |
| S42 terminal-blind obstruction | **PROVEN** | The named `prec_dagger` temporal selector when promised for every legal `sigma` but supplied no physical P5 bridge; candidate-own `sigma_dagger` trace |
| S43 reset charge barrier | **PROVEN** | Raw S22 episode counting in the broad total-exact per-pair grammar; two S coordinates exhaust but never exceed a reset budget |
| S43.1 colored two-proxy cut | **PROVEN** | Per-pair `K=2`, total exact owner fidelity, isometry fixed within the pair; adaptive candidate-own obstruction for every alleged-winning `sigma` |
| Full `G_A^{2/pair}` success class | **OPEN** | A candidate may change the total binding inside each episode; broader non-total, window-coded, or lagged branch-(A) repairs also survive outside this grammar |
| S44 no certificate discount | **PROVEN** | Fixed `E_S`, no physical reconciliation, completed nonwinning F/`Fhat` pairs, common winning-`sigma` phase, any Definition 39.1 selector |
| S44.1 S30 certificate barrier | **PROVEN AT CONDITIONAL SCOPE** | S30 abstract labeled checkpoint; post-F and post-next-first-placement variants |
| S45 exact S30 transversal | **PROVEN** | Full urgent family of S30's abstract label has `tau_E=5` |
| Global P0--P6 plus P5R coupling | **OPEN** | No theorem puts every alleged-winning `sigma` and every legal real-S continuation in `A_FS2^ET` or another complete branch system |
| `NL_F` | **OPEN** | D2 remains the logical bridge; neither determinacy alternative is selected |

There are no **SKETCH** or **CONJECTURE** results in this round.

### 49.2 Requested objective verdicts

1. **P3 carrier: POSITIVE-AT-SCOPE.** S40 is an exact causal carrier on
   `A_FS2^ET`. S41 proves the class physically nonempty while directly
   exercising S18 and a proxy-assisted terminal window. S42 proves that the
   named terminal-blind selector is insufficient. What remains open is the
   named universal quantifier: derive the terminal bridge and continued
   `A_FS2` membership for every alleged-winning `sigma`, or force their
   failure on that carrier's own winning-`sigma` history.
2. **Per-pair `K=2`: PARTIAL.** S43 proves that a reset escapes S32's raw
   counting. S43.1 nevertheless kills the exact subclass whose isometry is
   fixed within each pair. General intra-pair rebinding remains open.
3. **S30: NEGATIVE AT A CONDITIONAL MODULE SCOPE.** S44 proves that a
   common-phase deficit certificate cannot substitute for missed physical
   service. S45 sharpens the inherited three-window lower bound to exact
   `tau_E=5`. No strategy-specific reachability theorem for S30 is claimed.

**Most valuable new theorem.** S40 identifies the smallest genuine positive
P3 mechanism found so far: spatially unrelated legal events can carry the
nonterminal prescriptions, leaving physical co-terminal alignment as the
single F-role semantic bridge. S41 shows this is not formal relabeling by
passing the exact S18 and S20 stress phenomena on one append-only trace.

## 50. Authoritative obstacle and obligation ledgers

### 50.1 Round-5 review's twelve-item unresolved-obstacle ledger

This table uses the round-5 hostile review's list as the authoritative input
state. A scoped advance does not close the global row.

| # | Authoritative obstacle | Round-6 disposition |
|---:|---|---|
| 1 | Per-pair and broader branch-(A) repair | **PARTIAL.** S43 proves that lifetime counting cannot cross a pair reset. S43.1 refutes per-pair `K=2` when `T` is fixed within the pair. Intra-episode total rebinding, non-total/window recoding, and one repair per placement indefinitely remain **OPEN**. |
| 2 | Pre-checkpoint and recurring P3 transfer | **PROVEN on `A_FS2^ET`; OPEN globally.** S40 pairs every reached sequential prescription with canonical service. It assumes continued membership and the physical terminal bridge. |
| 3 | Coverage outside `A_FS2` | **PARTIAL negative.** S44/S44.1 exclude the natural common-phase deficit-certificate bridge at fixed debt when `tau_E>2`; S45 gives exact `tau_E=5` for S30. First-unsafe, terminal, old-certificate-failing, reconciled, and phase-lagged branches remain **OPEN**. |
| 4 | P5R through every lag/recode | **No global closure.** S34, S38, and the `A_FS2^ET` outer premise remain the proved modules. S43.1 uses P5R to close terminal opposite-role cuts. S14 and S25 remain binding outside guarded/certified traces. |
| 5 | F-service/P3 compatibility | **PROVEN on `A_FS2^ET`; OPEN universally.** The event carrier shows that coordinate equality is unnecessary and S41 exercises real service. Nothing derives co-terminal service for arbitrary alleged-winning `sigma`. |
| 6 | Reverse shadow-to-real legality | **PROVEN irrelevant for the event carrier; OPEN for inverse-map carriers.** S40 never requests `T^{-1}(z)`, and S41 has an actual legal shadow prescription with illegal inverse. S18 and S13 remain binding for point/FIFO schedules. |
| 7 | Shadow-`Fhat` terminal fidelity | **PROVEN on terminal-aligned traces; OPEN globally.** S41 transfers a proxy-assisted win; S42 refutes the named terminal-blind selector. No theorem forces alignment for arbitrary alleged-winning `sigma`. |
| 8 | Strategy domain and physical persistence | **PROVEN on the new finite class; OPEN globally.** Every S41 move is a genuine append; S40 retains and recomputes all stones. Arbitrary recodes and selectors still owe one legal `sigma` history. |
| 9 | Causality | **PROVEN for Definition 46.1; OPEN globally.** Service and prescriptions are selected after S's completed turn, and `z_2` only after the first event. Other outer carriers may still trigger S12. |
| 10 | Window-certificate maintenance | **PARTIAL negative.** S44 applies even to a dynamically reselected Definition 39.1 selector at the post-F checkpoint. Newly admitted debt, physical reconciliation, phase-lagged events, and broader certificates remain **OPEN**. |
| 11 | Permanent fencing | **OPEN; no construction.** S31's exact six-blocker geometric cost remains binding. S45 concerns urgent service at one labeled state, not installation of a permanent fence. |
| 12 | Strategy-specific reachability and outcome | **PARTIAL.** S42 is candidate-own but uses a legal strategy not proved winning. S43.1 is adaptive on each candidate-own alleged-winning-`sigma` checkpoint in its fixed-within-pair class. S30 remains abstract. The global coupling and `NL_F` remain **OPEN**. |

### 50.2 Round-4 review's ten-item agenda, carried forward

| Agenda item | Round-6 status | Exact advance and remaining duty |
|---:|---|---|
| 1. Pre-checkpoint P3 transfer | **PROVEN AT S40 SCOPE; OPEN globally** | Event pairing supplies a genuine recurring interface wherever canonical service and terminal alignment are available. Universal `A_FS2^ET` coverage is unproved. |
| 2. P2/P4 at each real-S coordinate | **PARTIAL** | S34 remains positive on `A_FS2`; S43.1 refutes pair-static total-exact repair. General branch selection remains open. |
| 3. P5R during every lag/recode | **PROVEN in inherited conditional modules; OPEN globally** | `A_FS2` shielding and S38 certification remain sound. S44 prevents a deficit selector from hiding missed urgent service. |
| 4. F-service compatibility | **PROVEN AT S40 SCOPE; OPEN globally** | S40 makes a legal temporal P3 pair from canonical service; S41 is active. The arbitrary-`sigma` terminal bridge and `tau_E>2` coverage are missing. |
| 5. Permanent-fence installation | **OPEN** | S31 unchanged; availability, interrupted installation, S occupation, and P3 remain unresolved. |
| 6. Reverse P3 legality | **PROVEN for event pairing; OPEN for spatial transfer** | The event carrier survives S18 by construction and witness. Any claimed inverse or FIFO map must still pass S18/S13. |
| 7. Shadow-`Fhat` terminal fidelity | **PROVEN AT CONDITIONAL SCOPE; OPEN globally** | S41 transfers one proxy-assisted terminal event; S42 proves that the named selector cannot drop the condition. |
| 8. Strategy domain and physical persistence | **PROVEN on new finite traces; OPEN globally** | Definition 46.1 uses one genuine history; every filler/proxy/service stone persists. Broader update systems remain unproved. |
| 9. Causality | **PROVEN locally; OPEN globally** | `K_E` and padding are post-S selectors, and no S action intervenes before F uses them. This does not certify every outer branch. |
| 10. Strategy-specific reachability and outcome | **PROVEN for S43.1's subclass; PARTIAL elsewhere** | The colored cut is adaptive on candidate data. S41/S42 do not supply a globally winning `sigma`; S30 has no forced shadow reachability. `NL_F` remains open. |

### 50.3 P0--P6/P5R cross-ledger

| Obligation | Status after round 6 | Binding disposition |
|---|---|---|
| `P0 STRATEGY-DOMAIN` | **PROVEN on `A_FS2^ET`; OPEN globally** | S40 queries only the reached genuine history; S41's on-path prescriptions extend to a total legal strategy. Membership for every alleged-winning strategy is not proved. |
| `P1 OPENING/CADENCE` | **PROVEN for cadence/legal-prefix and event traces** | S15 starts S41. Paired nonwinning events advance both phases identically; wins stop immediately. |
| `P2 REAL->SHADOW` | **PARTIAL** | In branch B, S34 handles admitted rolling S actions. Branch A pair-static repair is excluded. S40 concerns F-role events, not global P2 coverage. |
| `P3 SHADOW->REAL` | **PROVEN AT `A_FS2^ET` SCOPE; OPEN globally** | Temporal pairing discharges sequential legality without an inverse map. Physical terminal alignment and continued serviceability are class premises. |
| `P4 COLLISION` | **PARTIAL** | F-role cross-board collisions disappear in event pairing; S41's final inverse is already occupied. S43.1 exposes an unavoidable S-role opposite-color collision when `T` is pair-static. |
| `P5 SHADOW-F-TERMINAL` | **PROVEN AT TERMINAL-ALIGNED SCOPE; OPEN globally** | S41 transfers a proxy-assisted first-placement win. S42 refutes one named terminal-blind selector; inherited S20 still binds second-placement and universal semantics. |
| `P5R REAL-S-TERMINAL-REFLECTION` | **PROVEN in inherited guarded/certified classes; OPEN globally** | `C_shield`, `A_FS2`, and S38 remain exact. S43.1's terminal cut branch cannot be repaired against alleged-winning `sigma`; S14/S25 remain mandatory. |
| `P6 CAUSALITY` | **PROVEN for Definition 46.1; OPEN globally** | No service coordinate is chosen across an intervening S turn, and sequential queries respect engine timing. Other branches still owe S12. |

## 51. Hostile-review attack surface

### 51.1 Load-bearing limitations

1. **`A_FS2^ET` is a trace class, not a progress theorem.** S40 proves every
   reached event correct; it does not prove that every real-S coordinate is
   first-safe, that every old debt is certifiable, or that every urgent family
   has `tau_E<=2`.
2. **Terminal alignment is physical.** A temporal pair, equal phase, equal
   event index, or equal owner does not imply a real win. The real service
   append itself must complete a real F window when the paired shadow append
   completes an `Fhat` window.
3. **S41's strategy is not alleged-winning globally.** It proves a nonempty
   legal carrier cylinder and exact S18/S20 stress. It does not answer the
   quantifier over every globally winning `sigma`.
4. **S42 has a deliberately broader strategy domain and narrower selector
   scope.** Its negative control kills the named `prec_dagger` carrier when
   promised for every legal pure strategy. It cannot be cited against another
   selector or as a forced history for the unknown alleged-winning-only
   domain.
5. **Event pairing abandons spatial transfer.** This is why S18 cannot fire,
   but it also means that future real/shadow geometry is connected only by the
   rolling S certificate, real service, and the terminal bridge. No hidden
   pointwise F invariant may be inferred.
6. **Second placements remain sequential.** `z_2` is not queried if `z_1`
   wins, and `k_2` is not played if `k_1` wins. Every theorem uses the
   post-first physical boards.
7. **The S20 obligation is universal inside the class.** S41 happens to end
   on a first placement, but Definition 46.1 requires the same physical
   bridge for a second-placement terminal prescription.
8. **Per-pair S43.1 needs fixed `T` within the pair.** A new binding after
   `c_1` can rotate an endpoint of the persistent `p_S--p_F` edge out of the
   represented/proxy cut. That is the exact surviving total-exact mechanism.
9. **S43 is a method limitation, not a positive coupling.** Two available
   charges matching two coordinates says nothing about whether either repair
   can be performed legally or terminal-faithfully.
10. **S44 fixes physical debt through the F turn.** An actual correct-role
    shadow certificate may reconcile it; a label change may not. Phase-lagged
    and event-level certificates outside Definition 39.1 survive.
11. **S45 is an exact real-board census only.** Its `tau_E=5` sharpens S30's
    inherited `>=3` lower bound but does not manufacture a shadow history or
    force the label on a candidate's alleged-winning-`sigma` play.
12. **Every physical stone persists.** In particular S41's opener proxy,
    filler, off-line service stones, and `(-8,0)` prescription remain in all
    later support and terminal calculations. No proof uses undo, erasure,
    recoloring, or relocation.
13. **Branches remain locally alternative and globally conjunctive.** A
    candidate may choose (A), (B), or (C) for one observed S coordinate, but
    P3, service, persistence, P5/P5R, causality, and all regression tests are
    mandatory whichever branch is chosen.
14. **No outcome inflation.** Obstructing a carrier class does not prove that
    S has no winning strategy; constructing a conditional carrier does not
    prove its universal coverage. `NL_F` remains OPEN.

### 51.2 Binding regression matrix

| Regression | Round-6 treatment | Remaining boundary |
|---|---|---|
| S12 preannounced F coordinate | S40 selects service after the S pair and uses it before S moves again | Other outer branches may still preannounce |
| S13 fixed-isometry FIFO frontier | Event pairing is neither inverse copy nor FIFO point mapping | S13 remains binding on every schedule with its premises |
| S14 literal one-cell terminal lag | `A_FS2` shielding is inherited by S40 | Unguarded lag remains excluded |
| S18 proxy-supported reverse illegality | S40 does not invert; S41 has `z=(-8,0)` with illegal inverse `(-10,0)` | Every spatial P3 claim still owes reverse legality |
| S20 proxy-fabricated `Fhat` win | S41 transfers a proxy-assisted physical window; the named S42 selector fails without the bridge | Universal terminal alignment remains open |
| S25 older-surplus real-S win | Prevented only on inherited guarded/certified trace scopes | All other lag/recode branches still owe P5R memory |
| S30 high-transversal fork | S44 blocks the natural certificate escape; S45 proves exact `tau_E=5` | Strategy-specific reachability, reconciliation, and broader certificates remain open |
| S31 permanent fence | No new installation theorem | Exact cost six and all availability/timing issues remain binding |

## 52. Provenance and resume point

### 52.1 Provenance

**Input state.** Branch `hunt/gap-raw`, input HEAD
`3000a117d10a2148f744412aae26e053cf6babbc` (short `3000a117`). This authoring
pass creates no commit and does not amend, reset, or move a branch reference.
The only intended deliverable is `STRATEGY_STEALING_ROUND6.md`. Pre-existing
unrelated untracked workspace entries were left untouched.

**Required corpus read first, in order and in full.**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including binding section 35 and the
   corrected section-34.1 quantifiers, then
   `STRATEGY_STEALING_REVIEW_ROUND4.md`; and
5. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, then
   `STRATEGY_STEALING_REVIEW_ROUND5.md` and its authoritative twelve-item
   unresolved-obstacle list.

No `GAP_RAW_*` file was read or used as mathematical evidence. No claim in
this document depends on one.

**Rule sources read in full.**
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
The proofs use only the rooted opening, physical radius-eight support,
sequential insertion, immediate per-placement six detection, terminal
no-continuation rule, and append-only forward histories tied down in section
45.2.

**Machine work.** None. No Cargo command, Lean build, harness, executable
search, or proof search was run. All distance, cadence, window, and
transversal checks are hand proofs.

### 52.2 Sharpest next question [OPEN]

For every alleged-winning `sigma` and every rolling `A_FS2` prefix, can the
same event carrier **derive** rather than assume a co-terminal real-F window
for whichever first or second `Fhat` prescription ends the shadow game, while
keeping future debt certificates fresh and serviceable; or can terminal
misalignment be forced on that carrier's own legal winning-`sigma` history?

The first positive target is a causal physical F-window deficit invariant
maintained by the canonical service events. The first negative target is an
adaptive version of S42 in which the strategy is genuinely alleged-winning,
not merely a legal pure strategy chosen for a diagnostic cylinder.
