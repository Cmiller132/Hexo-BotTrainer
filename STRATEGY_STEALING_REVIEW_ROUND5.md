# R-ST5-REV — Hostile review of `STRATEGY_STEALING_ROUND5.md`

## Method and review boundary

**Reviewed artifact:** `STRATEGY_STEALING_ROUND5.md` at commit
`d0af2ef410b718474668be523e55e04ec6f818e1` on branch
`hunt/gap-raw`.

**Named input and concurrent advance:** the artifact names
`67f996d1df95f0cafaabdd87baa6659c2eb9f59f` as its authoring input.
Git metadata confirms the ancestry

`67f996d1 -> 110042e5 -> d0af2ef4`.

The intermediate commit is
`110042e5f8b563aa7c5587543182c15ffcd0afa9`. A name-only comparison
from the named input to that intermediate commit lists exactly
`.codex-gr/prompt-tempo-round6.txt` and
`GAP_RAW_PROOF_ROUND6.md`. The required strategy-stealing corpus and
the six cited engine files are unchanged across that advance. The reviewed
worktree copy of the round-5 artifact matches its blob at `d0af2ef4`.

**Required reading completed in order and in full:**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including its folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md`, including its folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including binding section 35 and
   the corrected section-34.1 quantifiers, then
   `STRATEGY_STEALING_REVIEW_ROUND4.md`; and
5. `STRATEGY_STEALING_ROUND5.md`.

I then read and checked the cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
This was a first-principles proof and source audit. I ran no Cargo command,
Lean build, harness, executable search, or proof search. I did not read or
modify any `GAP_RAW_*` file. File-name metadata was used only to verify
the provenance claim; no such file supplied mathematical evidence.

All cadence, support, coordinate, window, deficit, transversal, and terminal
checks below were recomputed by hand.

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** No theorem is refuted and no
MAJOR defect was found. S32 proves the lifetime-budget obstruction at exactly
its total-exact scope; S34 proves a genuinely finite, conditional guarded-lag
invariant; and S36--S39.1 establish the proposed physical window-certificate
module. The repairs concern a syntax/trace mismatch, two omitted handler
micro-cases, one overbroad P3 label, two inherited regressions missing from the
ledger, and a local source citation.

## Numbered findings

### 1. NOTE — the production rule contract used by round 5 is confirmed

> “A normal placement is legal exactly when it is empty and belongs to the
> physical radius-eight legal store.”

The cited sources establish the rule facts used in every proof:

- `coord.rs:76-95` implements
  `max(|dq|,|dr|,|dq+dr|)` and enumerates the inclusive radius ball;
- `legal.rs:17-18,123-145` fixes radius eight, removes the occupied
  coordinate, and inserts every physically empty cell in the new halo;
- `rules.rs:11-44` rejects terminal play, enforces the rooted opening,
  rejects occupancy, and tests the maintained legal set at both normal phases;
- `board.rs:83-105` inserts occupancy before updating legality and
  windows;
- `state.rs:149-160,289-357` gives the initial owner/phase, appends one
  stone, tests the win, and only then advances
  `Opening -> FirstStone -> SecondStone -> FirstStone`;
- `state.rs:203-252` exposes no terminal continuation; and
- `tactics.rs:13-17,21-75,205-208,451-485` gives the three Q/R/QR
  length-six families, the eighteen incident windows, and the all-six
  predicate.

The executable carrier is `i16`, while the general coupling theorems
retain the inherited `Z^2` scope. The explicit witnesses in sections 38
and 39 stay within hex norm eight and have update halos within norm sixteen.

**Proposed repair:** add `tactics.rs:21-75` to section 36.2's local
source list, because the later window constructions use the actual Q/R/QR
axis vectors. The facts themselves are correct.

### 2. NOTE — S32's three support-cut checkpoints have the claimed counts and phases

> “after two successful episodes and a nonterminal second `sigma` pair,
> success requires a third S-coordinate-reactive episode.”

The complete hand bookkeeping is:

| Checkpoint | Real `(F,S)` | Shadow `(Fhat,Shat)` | Actor/phase | `|A|` | `|P|` |
|---|---:|---:|---|---:|---:|
| S15 synchronization | `(1,2)` | `(2,3)` | F role `FirstStone` | 3 | 2 |
| first actual `sigma` pair | `(3,2)` | `(4,3)` | S role `FirstStone` | 5 | 2 |
| `c_1` plus first episode | `(3,3)` | `(4,4)` | S role `SecondStone` | 6 | 2 |
| `c_2` plus second episode | `(3,4)` | `(4,5)` | F role `FirstStone` | 7 | 2 |
| next `sigma` first placement | `(4,4)` | `(5,5)` | F role `SecondStone` | 8 | 2 |
| next `sigma` second, nonwinning | `(5,4)` | `(6,5)` | S role `FirstStone` | 9 | 2 |

At every restored total-exact checkpoint, injectivity maps every physical real
stone into `A`. Owner fidelity then leaves exactly one unmatched
`Fhat` stone and one unmatched `Shat` stone. Thus both `A`
and `P` remain nonempty after either escape, however the candidate
changes the isometry or partition. The genuine shadow support graph remains
connected. All of S22's folded round-4 premises therefore hold at each of the
three applications.

S22 makes every `c_i` real-empty and radius-eight supported. Once
`c_1` is placed, the second cut's empty `c_2` is automatically
different from the stored first coordinate, so it is legal at
`SecondStone`. The three real placements raise S only to three, four,
and five total stones; none can win. The first two associated shadow appends
raise `Shat` only to four and five.

**Proposed repair:** none. The second escape cannot repartition away the
third cut's nonempty-part or common-live premises.

### 3. NOTE — the intervening terminal fork in S32 is exhaustive

> “If `Fhat` wins with its sixth physical stone, the verdict cannot
> transfer ... It remains to consider the nonwinning second prescription.”

After (37.2), the first member of the next prescribed pair gives real F four
stones and shadow `Fhat` five, so neither side can win. The second gives
real F only five stones but `Fhat` six. Hence:

- if the shadow placement wins, real F has too few stones for any physical
  six certificate and the append-only shadow is terminal; Definition 37.1(6)
  fails if the candidate calls this a transferred F win; or
- if it does not win, both histories are normal and common-live at (37.3),
  so S22 supplies `c_3`.

At that genuine `sigma`-consistent `Shat` decision node, any
legal one-append `Shat` win would extend to a total counterstrategy and
contradict the assumption that `sigma` wins. Therefore branch (C)
cannot hide the third occupied target. A third reactive episode, or departure
from the grammar, is forced.

**Proposed repair:** none. This also respects S20's terminal warning rather
than continuing through a terminal shadow state.

### 4. MINOR — `G_A^2` nonemptiness is witnessed by a trace, not an algorithm

> “The grammar is nonempty [PROVEN]: the legal backing/filler partial
> execution ... is one such partial execution.”

Definition 37.1 makes `G_A^2` a family of deterministic causal update
algorithms. The cited round-4 object is a legal partial execution. It proves
that the proposed backing/filler syntax can execute on one history, but a
trace is not itself an algorithm.

**Proposed repair:** specify a deterministic candidate that follows the
folded backing/filler behavior on that history, uses the already required
fixed enumerations for all tie-breaks, and reports the first failed promise
off that path. This repairs grammar nonemptiness and does not affect S32 or
the emptiness of the extensional success class.

### 5. NOTE — lifetime `K=2` and per-pair `K=2` are kept distinct

> “S32 refutes a fixed total budget of two. It does not refute two reactive
> episodes per S pair.”

The proof correctly waits through the next actual F turn before selecting
`c_3`; S is not entitled to a third coordinate immediately after
`c_2`. Sections 37.3, 40, 41, and 42 consistently call the budget
lifetime/total and explicitly leave a reset of two episodes on every S pair
open.

**Proposed repair:** none. Per-pair `K=2` remains **OPEN**.

### 6. MINOR — S33/S34 skip the transient old-debt calculation

> “Each rolling step physically certifies the old singleton debt, so its
> E-live family disappears.”

The physical order is real `S@y` first and shadow
`Shat@T(e)` second. Between those appends, the old debt `e` has
not yet been physically reconciled and the new real cell `y` is also
unmatched. The proof jumps directly to the state after reconciliation.

The omitted calculation closes. For every old E-live window,

`delta' - m' = (delta - 1_{y in W}) - (m-1)
                 = (delta-m) + 1 - 1_{y in W} > 0.`

Thus no old-debt window can become terminal in the transient physical real
state. The first-safe or nonwinning test handles every new window through
`y`, after which the actual shadow append removes `e` from the
debt set.

**Proposed repair:** insert this microstep before saying that the old E-live
family disappears. S33 and S34 remain proved.

### 7. MINOR — the queue does not classify a filler-created `Shat` win

> “A legal `Shat` win during reconciliation ends the physical module.”

When `E_S=empty`, Definition 38.1 appends an enumerated legal filler
rather than `T(e)`. That filler is a physical `Shat` placement
and can in principle complete a shadow six. The engine then forbids any
continuation, but the definition's terminal sentence and
`A_FS2` clause 1 mention only a win on the old-debt certificate.

This omission does not break the real-board shield: the associated real
coordinate is still subject to first-safety or second-coordinate
nonwinning, and a physical `Shat` win is a safe terminal event once an
outer `sigma`-consistent carrier exists. It does leave the handler's
terminal grammar incomplete.

**Proposed repair:** say that *any* branch-(B) `Shat` append, whether
reconciliation or filler, ends the physical module when it wins. Broaden
`A_FS2` clause 1 accordingly and retain the existing warning that the
win contradicts `sigma` only under an outer P0/P3 carrier.

### 8. NOTE — the `A_FS2` restriction is finite, conditional, nonempty, and honestly stated

> “let `A_FS2` be the class of finite physical coupled trace segments”

> “`S30` and any other `tau_E>2` service state lie outside
> `A_FS2`.”

The conclusion really is restricted to finite handler-generated segments
whose own prefixes satisfy all six clauses. It proves neither:

- an infinite continuation;
- progress from every shielded state;
- a complete response to every legal S coordinate;
- a trace generated by an alleged-winning `sigma`; nor
- a P3 map from arbitrary prescriptions to the selected service cells.

Those limitations are repeated in S34, its P3 composition boundary, sections
41--42, and the ledgers. First-unsafe coordinates, terminal real-S
coordinates, old-debt certificate failures, and `tau_E>2` states are
excluded and delegated to branch (A), branch (C), or an unproved outer repair.
In particular, round 4's S30 fork fails clause 4; the source does not claim
that two-stone service survives it.

S35 nevertheless makes the finite class genuinely nonempty at its advertised
mechanism scope: it includes physical reconciliation, a nonempty rolling
debt, and an active service state with `tau_E=1`. This is stronger than
an empty-queue state witness but appropriately weaker than a global success
class.

**Proposed repair:** none to the theorem. Preserve “finite conditional trace
segment” whenever S34 is summarized.

### 9. NOTE — S33/S34's admission and two-stone service inequalities are exact

> “A new first-coordinate lag is shield-admissible exactly when ... 
> `delta>=2`.”

> “Every nonwinning second-coordinate lag is shield-admissible.”

After a first placement, `m=1`, so strict shielding is
`delta>1`, equivalently integer `delta>=2`. After a second
placement, `m=0`. A new E-live window has
`delta>=1` exactly when the placement did not complete a real S six.

At the following F checkpoint, every urgent E-live window has one or two
physically empty holes. A minimum transversal `K_E` with size at most
two hits each hole set. Every selected cell is within line distance at most
two of physical S support and is therefore legal at radius eight. If the first
service placement wins for F, the stop is sound. Otherwise the pair, padded
when necessary, blocks every urgent window. Every unblocked E-live window then
has `delta>=3>2` before the next S `FirstStone`.

This proves an exact two-placement real-board service rule, except for the
engine-mandated earlier F-win stop. It does not identify that pair with an
arbitrary prescribed `Fhat` pair, as the artifact correctly says.

**Proposed repair:** none.

### 10. NOTE — S35's coordinates, window census, and cadence all recompute

> “At the resulting F checkpoint the only urgent E-live window is
> `W={(0,1),...,(0,6)}`.”

The opening and transferred pair are legal under
`T(q,r)=(q+2,r)`. On the rolling S turn:

- real `e=(0,3)` is adjacent to the existing S chain, while filler
  `(0,1)` is adjacent to the shadow opening;
- the post-first real board has at most three S stones in every window through
  `e`, so `delta>=3>1`;
- real `y=(0,4)` is nonwinning, and shadow
  `T(e)=(2,3)` is fresh and adjacent to the certified chain.

At the F checkpoint, real S owns exactly
`(0,1),(0,2),(0,3),(0,4)`. Of the R-axis intervals through
`(0,4)`, the starts at `r=-1` and `r=0` contain the
blocker `F@(0,0)`; the start at `r=1` is the sole unblocked
deficit-two interval, with holes `(0,5),(0,6)`; later starts have
deficit at least three. Q and QR windows through the debt contain at most one
S stone. Hence `tau_E=1`.

Real service `(0,5),(1,5)` and shadow pair
`(2,5),(3,5)` are fresh, supported, and nonwinning. Real F then has
five total stones; shadow `Fhat` has six total but a longest relevant
run of four. The stated next rotation
`(1,4),(2,4) / (2,4),(3,4)` is also legal and leaves debt
`(2,4)` with no urgent window, so `tau_E=0`.

**Proposed repair:** none. The largest displayed hex norm is eight and the
largest update halo norm is sixteen.

### 11. NOTE — Definition 39.1 is a well-formed physical certificate over all engine axes

> “A physical window-deficit certificate is a history-causal selector
> `nu: U_E -> U_H` such that
> `delta_H(nu(W)) <= delta_R(W)`.”

`W_6` is the inherited family of every length-six Q, R, and QR window;
it is closed under translation and D6. Because `E_S` is finite and
each cell meets exactly eighteen engine windows, `U_E` is finite, with
`|U_E|<=18|E_S|`. The codomain `U_H` is generally infinite on
`Z^2`, but countable, so the stipulated global window enumeration makes
causal tie-breaking meaningful.

For `V in U_H`, the absence of `Fhat` stones means every cell
not counted as `Shat` is physically empty; there is no third owner.
Thus `delta_H` is a true physical-hole count. Radius eight is not part
of the static inequality, but it governs every actual append and the hole
filling used in S39.

The definition deliberately delegates honest construction/removal of
`E_S`, common-only real wins, cadence, P0, and P3 to the outer coupling.
It is a P5R module, not a full representation of the game.

**Proposed repair:** none. Do not detach the phrase “window-faithful
representation” from this conditional module boundary.

### 12. NOTE — S36 is genuinely outside every injective point representation

> “The constant selector `nu(W)=V` is ... not induced by any injective
> point map.”

For `e=(0,5)`, the eighteen incident real windows have the following
exact census:

- one R-axis window, `q=0,r=0..5`, is blocked by real
  `F@(0,0)` and is not in `U_E`;
- among the six Q-axis windows, five also contain `S@(1,5)` and have
  `delta_R=4`, while the remaining one has deficit five; and
- the other five unblocked R windows and all six QR windows contain only
  `e` and have deficit five.

Thus `|U_E|=17` and every member has deficit at least four. The shadow
window

`V={(0,0),(0,1),(0,2),(0,3),(0,4),(0,5)}`

contains exactly the first three cells as physical `Shat` stones and
no `Fhat` stone, so `delta_H(V)=3`. The constant assignment is a
certificate.

Both
`W_Q={(0,5),(1,5),...,(5,5)}` and
`W_R={(0,1),(0,2),...,(0,6)}` are E-live and distinct; they intersect
only at `e`, yet both are assigned to `V`. For an injection
`f`, equality `f[W_Q]=f[W_R]` would imply
`W_Q=W_R`, a contradiction. No injection, and therefore no
translation/D6 isometry, induces this selector.

All listed placements in both prefixes are fresh and within radius eight of
the origin; the owner cadence ends at common S-role `FirstStone`. The
real owners have four S and five F stones; shadow `Shat` has five and
`Fhat` has six split into runs of at most four. Both histories are
nonterminal.

**Proposed repair:** none. Many-to-one window assignment is not a noninjective
stone merge, so neither S22 nor S8.1 applies.

### 13. NOTE — S37's maintenance algebra is exact at its fixed-selector scope

> “`s'(W)=s(W)-1_{y in W}+1_{z in nu(W)}`.”

With `E_S` fixed, the E-live domain does not change during an S append;
with no `Fhat` append, the selected shadow windows remain unblocked.
The real append reduces `delta_R(W)` exactly when `y in W`, and
the legal, previously empty shadow append reduces
`delta_H(nu(W))` exactly when `z in nu(W)`. Subtraction gives
(39.4).

If a window is affected but has positive slack, one unmatched real decrement
is harmless. If it is tight, the shadow decrement is necessary, and because
`nu(W)` is `Fhat`-unblocked and `z` is legal, this means
exactly that `z` belongs to its physical hole set. One append services
all tight affected certificates precisely when their hole-set intersection
contains the chosen legal coordinate.

**Proposed repair:** none. The source correctly excludes dynamic reselection
of `nu` and newly admitting `y` to `E_S` from this lemma.

### 14. NOTE — S38 is the S26 same-step terminal route, and S39 does not conflict with S34

> “every real S win whose window meets current `E_S` produces a
> physical shadow-`Shat` win no later on that coupled step.”

For a terminal E-meeting real window `W`, the post-append state has
`delta_R(W)=0`. Certification forces

`0 <= delta_H(nu(W)) <= delta_R(W)=0`.

Thus the selected, physically `Fhat`-unblocked shadow window is filled
by six actual `Shat` stones. At the preceding live shadow state it
could not already have been full; the associated physical append is the only
new `Shat` occupancy. It therefore fills the last hole on that step.
This is exactly S26's actual terminal-certificate alternative, not a
post-terminal relabeling. Common-only real wins remain an outer coupling
obligation, as Definition 39.1 states.

For S39, if a live `Shat` node has an unblocked shadow window with
`k=delta_H(V)<=m`, then `k` is one or two. The window already
contains at least four `Shat` stones. Every hole is empty and at hex
distance at most five along the line from physical support, hence legal under
radius eight. `Shat` fills the one or two holes during its current
turn, contradicting a winning `sigma`. Therefore
`delta_H(V)>m`, and any deficit certificate gives
`delta_R(W)>=delta_H(nu(W))>m`.

This is consistent with section 38: `A_FS2` already admits only
shield-safe debt. Moreover its clause-6 `Fhat` pair is not generally
prescribed by `sigma`, so S39's strategy-domain premise need not hold
until the open P0/P3 interface is supplied.

**Proposed repair:** none. S38 is transfer conditional on maintenance; S37 is
only one local maintenance test; S39.1 does not collapse phase-lagged or
event-level certificates.

### 15. MINOR — S35 is “P3-shaped,” not an advance for arbitrary winning `sigma`

> “S35 gives one legal P3-compatible service trace.”

> “Agenda {1 locally,3,4,8}.”

The displayed shadow pair is a legal pair associated coordinatewise with the
real service pair, and its on-branch choices extend to some legal pure
strategy. The artifact expressly says that strategy is not claimed winning.
Round 4's agenda item 1, however, quantifies over every alleged-winning
`sigma`. “P3-compatible” can therefore be read more strongly than the
witness proves.

**Proposed repair:** call S35 a “P3-shaped, phase/legality-paired physical
trace” and credit it to the geometry side of items 3--4 and physical
persistence in item 8. Reserve an item-1 advance for S32's negative crossing
of actual `sigma` pairs; arbitrary positive P3 composition remains
**OPEN**, as the substantive ledger already says.

### 16. MINOR — the claimed complete regression ledger omits inherited S13 and S14

> “Sections 40--42 update every item in the authoritative ten-item agenda and
> retain all inherited regressions.”

The ten numbered rows do map one-for-one to the corrected round-4 agenda, and
they retain S18, S20, S12, S25, S30, and S31. Round 5 never names:

- round-2 S13, the fixed-isometry one-stone FIFO frontier failure; or
- round-2 S14, the literal one-S-stone-lag terminal-count failure.

Their subject matter is not silently solved. The rolling queue differs from
S13 by using persistent genuine fillers/proxies and physical reconciliation,
and S34 avoids S14 only on shield-admitted traces. S13 remains a test for a
fixed-isometry FIFO interpretation of lag; S14 remains a test for any
unguarded literal lag. The categorical OPEN rows cover them, but the literal
claim to retain *all* regressions needs the named cross-reference.

**Proposed repair:** add S13 to agenda items 2/6 and S14 to item 3 and the
P5R row, stating why `A_FS2` is outside the former and conditionally
passes the latter. No theorem status changes.

### 17. NOTE — the corrected agenda quantifiers, `NL_F` boundary, and provenance are otherwise accurate

> “Checkpoint (28.1) is conditional on a successful total-exact P3 transfer.”

> “No arbitrary winning strategy is refuted; `NL_F` stays open.”

Section 40 preserves the round-4 section-35 structure: per real-S coordinate,
branch (A), (B), or (C) is an alternative, while service, persistence, and
regression obligations remain mandatory in every branch. S32's negative
coordinates are selected from the candidate's own successive committed
bindings around actual `sigma` pairs. S35 and S36 are called reachable
mechanism states rather than strategy-forced negative gadgets. The
conditional status of (28.1), the finite horizon of `A_FS2`, and the
absence of a global P0--P6/P5R conjunction are retained.

The document invokes D2 only as the established determinacy bridge. It never
claims that the arbitrary alleged-winning strategy has been refuted, and it
marks `NL_F` OPEN in the header, ledgers, objective verdicts, and
provenance section.

Git confirms the named ancestry and external name-only change described in
the method preamble. A textual audit finds no theorem, coordinate, lemma, or
citation derived from a `GAP_RAW_*` artifact; the only occurrences are
the provenance filename and no-edit statement. Git cannot prove the historical
mental fact that an author never opened a file, but the reviewed artifact has
no evidentiary dependence on one.

**Proposed repair:** none beyond Findings 1, 15, and 16.

## Per-theorem verdicts

| Result | Source status | Review verdict | Exact disposition |
|---|---|---|---|
| Production rule contract | PROVEN | **CONFIRMED-WITH-MINOR-ERRATA** | All predicates and timing match production; add the local Q/R/QR axis range from Finding 1 |
| Definition 37.1 candidate/success classes | Definition plus grammar nonemptiness PROVEN | **CONFIRMED-WITH-MINOR-ERRATA** | Scope is coherent; replace the trace-as-algorithm nonemptiness witness |
| S32 lifetime-two-escape obstruction | PROVEN | **CONFIRMED** | All three cut applications meet S22's normal/common-live/nonempty-part premises; the terminal fork is exhaustive |
| S32 own-history reachability | PROVEN | **CONFIRMED** | `c_1,c_2,c_3` are selected from successive committed bindings around actual `sigma` pairs |
| Section 37.3 scope boundary | PROVEN | **CONFIRMED** | Fixed lifetime/total `K=2` only; per-pair `K=2` correctly remains OPEN |
| Definition 38.1 rolling queue | Definition | **CONFIRMED-WITH-MINOR-ERRATA** | Physical singleton debt rotation is coherent; classify a filler-created `Shat` win |
| Definition 38.2 `A_FS2` | Definition | **CONFIRMED** | Nonempty class of finite conditional trace segments; first-unsafe, terminal, certificate-failing, and `tau_E>2` continuations excluded |
| S33 phase-exact admission/service | PROVEN | **CONFIRMED-WITH-MINOR-ERRATA** | Inequalities and transversal service exact; add the transient old-debt calculation |
| S34 rolling guarded-lag invariant | PROVEN at scope | **CONFIRMED-WITH-MINOR-ERRATA** | Preventive P5R and real-board service hold on every finite `A_FS2` segment; no global progress or P3 theorem |
| S34 P3 composition | OPEN | **CONFIRMED OPEN** | Arbitrary `sigma`, S18, S20, and same-step P5 remain unresolved |
| S35 active-service witness | PROVEN | **CONFIRMED-WITH-MINOR-ERRATA** | Coordinates, `tau_E=1`, service, and rotation are exact; call the trace P3-shaped, not arbitrary-winning-`sigma` compatible |
| Definition 39.1 deficit certificates | Definition | **CONFIRMED AT CONDITIONAL MODULE SCOPE** | Physical Q/R/QR deficit cover, no cell map, outer honesty and common-only wins delegated |
| S36 non-point-induced witness | PROVEN | **CONFIRMED** | Seventeen E-live windows, physical deficit-three cover, and injectivity contradiction all exact |
| S37 fixed-selector maintenance | PROVEN | **CONFIRMED** | Equation and common-hole intersection criterion exact for fixed `E_S,nu` |
| S38 deficit-certificate P5R | PROVEN at scope | **CONFIRMED** | Implements S26's same-step physical terminal certificate on complete maintained traces |
| S39 winning-strategy barrier | PROVEN | **CONFIRMED** | One/two physical holes can be filled in the current `Shat` turn |
| S39.1 deadline obstruction | PROVEN | **CONFIRMED** | Implies `delta_R>m` at common-phase winning-`sigma` nodes; no contradiction with S34 |
| Section 40 ten-item mapping | PROVEN ledger maintenance | **CONFIRMED-WITH-MINOR-ERRATA** | Correct A/B/C quantifiers and statuses; explicitly restore inherited S13/S14 |
| Provenance | factual record | **CONFIRMED AT REPOSITORY-EVIDENCE SCOPE** | Hash ancestry and name-only external change agree; no textual GAP_RAW evidentiary use |
| Global P0--P6 plus P5R coupling | OPEN | **CONFIRMED OPEN** | No single mechanism covers every alleged-winning `sigma` |
| `NL_F` | OPEN | **CONFIRMED OPEN** | D2 is only a bridge; neither determinacy alternative is selected |

## Exact unresolved obstacles after review

1. **Per-pair and broader branch-(A) repair.** S32 excludes only a lifetime
   total-exact isometric budget of two. Two episodes per S pair, one repair
   per placement indefinitely, non-total/window recoding, and nonisometric
   zero-lag representations remain open.
2. **Pre-checkpoint and recurring P3 transfer.** For every alleged-winning
   `sigma`, realize each sequential `Fhat` prescription as a
   legal real-F action with a genuine restored invariant, or give an earlier
   recode, lag, or sound terminal repair. S35 is only a legal P3-shaped trace.
3. **Coverage outside `A_FS2`.** First-unsafe real-S coordinates,
   terminal coordinates, unavailable old-debt certificates, and
   `tau_E>2` service states need branch (A), branch (C), reconciliation,
   or a different invariant. S30 remains an explicit three-axis test.
4. **P5R through every lag/recode.** Every real-only S cell must retain
   `delta>m`, become physically certified, receive a physical F blocker,
   or obtain an actual same-step shadow-`Shat` certificate. S14 and S25
   remain mandatory terminal-memory regressions.
5. **F-service/P3 compatibility.** The canonical `K_E` pair is selected
   after the S turn and is legal on the real board when `tau_E<=2`.
   Nothing proves that arbitrary `sigma` prescriptions can be causally
   paired with it while preserving future geometry.
6. **Reverse shadow-to-real legality.** S18's proxy-supported reply and all
   sequentially updated unsupported/collision sets remain unresolved.
   S13 also remains binding on its fixed-isometry FIFO schedule.
7. **Shadow-`Fhat` terminal fidelity.** S20's proxy-assisted
   second-placement win is not repaired. S32's six-versus-five branch gives
   the same exact need for a physical real-F certificate.
8. **Strategy domain and physical persistence.** Every filler, proxy, queue
   rotation, selector change, and recode must remain one genuine legal,
   append-only, `sigma`-consistent shadow history. Old physical stones
   keep all support, occupancy, blocking, and window effects.
9. **Causality.** A future real-F coordinate may not be fixed across an
   intervening S turn under S12's premises. Selecting `K_E` after that
   S turn solves only the local service selector, not the outer carrier.
10. **Window-certificate maintenance.** S37 solves one fixed-selector step.
    Dynamic selector changes, newly admitted debt, phase-lagged event
    certificates, common-only real wins, and simultaneous legality/terminal
    obligations still need a universal physical handler.
11. **Permanent fencing.** S31's six blockers are the exact unconstrained
    geometric cost, but cell availability, interrupted installation, S
    occupation, and P3 compatibility remain open.
12. **Strategy-specific reachability and outcome.** A negative gadget must be
    forced on the candidate's own legal `sigma`-consistent history.
    S35/S36 are reachable diagnostic examples, not forced winning-strategy
    states. Until every alleged-winning `sigma` is refuted or one global
    carrier is built, `NL_F` remains OPEN.

## Overall verdict and objective dispositions

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** There is no REFUTED or MAJOR
finding. The most consequential repair is definitional rather than
mathematical: branch (B) must classify a filler-created physical
`Shat` win, and its proof should expose the transient two-debt
microstate. Neither issue changes S34's finite conditional invariant.

1. **Section 37 objective — CONFIRMED.** The third support cut really applies
   after two successful escapes; count-preserving repartition cannot empty the
   proxy side. The theorem excludes lifetime
   `C_A^{K=2,total}` only. Per-pair `K=2` remains OPEN.
2. **Section 38 objective — CONFIRMED-WITH-MINOR-ERRATA.** S33/S34 prove the
   guarded rolling queue and exact real-board two-stone service on finite
   `A_FS2` segments. The class is nonempty, excludes S30's
   `tau_E=3` fork explicitly, and supplies no arbitrary-`sigma`
   P3 carrier.
3. **Section 39 objective — CONFIRMED.** The deficit certificate is physical
   and well formed over all engine windows; S36 is not induced by any
   injection or isometry; S37's algebra and S38's S26 transfer are exact; and
   S39.1 is a scope limitation consistent with, not contradictory to, S34.
4. **Section 40 ledger objective — CONFIRMED-WITH-MINOR-ERRATA.** The
   authoritative ten agenda items and corrected quantifiers are preserved,
   but S13/S14 should be named before the ledger claims to retain every
   inherited regression.

