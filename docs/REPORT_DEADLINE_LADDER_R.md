# Deadline ladder Phase R: the global clock has no sound finite rung

## Executive verdict

**MEASURED — NO-GO.** No tested horizon-indexed necessary condition dismisses
any of the 248 grind roots at the requested sound deadline, and therefore none
dismisses any of the 599.491 seconds of measured grind Unknown wall.  The two
finite lower bounds have substantial bite at `h=2/4/8`, but both reach zero by
`h=16`; a global-fill instantiation would be more than 4.294 billion
placements away.

**CODE-FACT — premise failure.** More importantly, the rules board has no
finite global-fill deadline.  It is explicitly an unlimited sparse board over
axial coordinates ([`coord.rs`, lines 1–15](../packages/hexo_engine/rust/src/coord.rs#L1),
[`board.rs`, lines 1–20](../packages/hexo_engine/rust/src/board.rs#L1), and
[`state.rs`, lines 97–105](../packages/hexo_engine/rust/src/state.rs#L97)).
The `i16 × i16` carrier has `2^32` representations, but it is an implementation
encoding, not the Lean `Z²` game board.  The inherited report had already
pre-registered this exact boundary: filling the finite Rust carrier is
incompatible with the Lean semantics
([`RESEARCH_DIVERGENCE_1.md`, lines 919–921](RESEARCH_DIVERGENCE_1.md#L919)).
Consequently `D(P)` is **infinite/undefined**, not a finite sound deadline, for
every position measured here.

**MEASURED.** Exact `NoAdmissibleFirstTurn` is broader than
`NoJointCarrier`—54.22% versus 29.94% of self-play fresh-turn roots and 51.74%
versus 13.77% of human fresh-turn roots—with zero hits on 2,941 known WINs.
This does not pass the GO bar: it is the already-computed empty-expansion fact,
not a horizon-indexed defender invariant or an absolute no-win theorem.  It
hits zero grinds and at most 0.879% of the available Unknown-wall proxy.
Counting this known exact control as an `h=∞` ladder member would be a category
error and would contradict the requested consumption target.

**HYPOTHESIS — recommendation.** Run the reduced Lean program only: exact
`h=2` current-turn completion, exact `h≤4` standing-threat/double-threat
classification, and the rank-two boundary-pair quotient.  Do not fund the
parametric horizon induction.

## 1. Scope, semantics, and artifacts

**MEASURED.** The analyzer is
[`deadline_ladder_r.py`](../.scratch/deadline_ladder_r.py); its complete output
is [`deadline_ladder_r.json`](../.scratch/deadline_ladder_r.json).  It scans the
required 3,255 self-play, 2,720 human, 468 puzzle, 248 grind, and 19 forcing
roots.  It independently reconstructs length-six windows, phase ownership,
hitting numbers through two, exact pair admission, and the two shallow
predicates.  The engine cross-check replayed 6,294 unique roots through the
public threat diagnostic and found 0 mismatches.

**CODE-FACT.** A physical ply in this report is one placement.  The opening is
one Player-0 placement; thereafter `FirstStone` leaves the same player to make
`SecondStone`, and only the second placement passes control
([`state.rs`, lines 1–9](../packages/hexo_engine/rust/src/state.rs#L1) and
[`state.rs`, lines 318–334](../packages/hexo_engine/rust/src/state.rs#L318)).
For a move list of length `n>0`, the mover is Player 1 when
`floor((n-1)/2)` is even and Player 0 otherwise; phase is `FirstStone` when
`n-1` is even and `SecondStone` otherwise.  The current-turn budget is two at
`FirstStone` and one at Opening/`SecondStone`
([`threats_shared.rs`, lines 46–53](../packages/hexo_models/rust/src/threats_shared.rs#L46)).

**MEASURED — evidence boundary.** Grind and forcing wall use every Unknown row
in `raws/lanec_labels.jsonl`.  Full per-root wall raws for all 3,255/2,720/468
roots are not present in this worktree, so their wall-weighted figures use the
current production-shaped `main4_integration_gate2` dev records: 325
self-play, 275 human, and 36 puzzle Unknowns.  Those are explicitly sample
proxies, not full-cohort estimates.

**CODE-FACT — node-level boundary.** The public batch/probe API returns root
verdict and aggregate counters, not visited interior states.  A genuine
per-node prevalence probe would require new `cfg(test)` Rust telemetry, which
the brief permits only if unavoidable and the inherited analyzers show is not
needed to decide this kill test.  The measurements below are therefore
root-level.  They can understate a condition born deeper or overstate one whose
root hit lies off the selected PN path.  Existing real-solve evidence already
shows every `NoJointCarrier` hit terminates at the one-expansion/two-node
boundary, so it saves certificate representation rather than target search
wall (`RESEARCH_DIVERGENCE_1.md` §7.1).

## 2. Deadline landscape

### 2.1 The sound deadline

**CODE-FACT.** `HexCoord` has two `i16` lanes, and `PackedCoord = u32` packs
their offset bit patterns bijectively
([`legal.rs`, lines 22–40](../packages/hexo_engine/rust/src/legal.rs#L22)).
That yields an implementation-carrier count of `65,536² = 4,294,967,296`
representations.  It does **not** turn the documented unlimited board into a
bounded rules board.  Coordinate addition/distance is ordinary `i16`
arithmetic, and the state clock itself is `u32` incremented with `+= 1`
([`state.rs`, lines 97–106](../packages/hexo_engine/rust/src/state.rs#L97),
[`state.rs`, line 305](../packages/hexo_engine/rust/src/state.rs#L305)).  A full
carrier cannot be reached or named as a valid absolute semantic horizon:
`2^32` is already one beyond `u32::MAX`.

**CODE-FACT.** Thus the requested rules-semantic quantity is:

`D(P) = placements remaining until board fill = ∞`

for every cohort row.  No finite `D` distribution exists.

### 2.2 Diagnostic packed-carrier surrogate

**MEASURED — not a theorem clock.** To quantify how far away the owner's
suggested clock would be even under the rejected carrier interpretation, the
analyzer also reports `D_i16(P)=2^32-|stones(P)|`:

| cohort | n | placements min / p50 / p90 / max | `D_i16` min / p50 / max |
|---|---:|---:|---:|
| 248 grinds | 248 | 13 / 57 / 79 / 85 | 4,294,967,211 / 4,294,967,239 / 4,294,967,283 |
| `human_v1` | 2,720 | 8 / 24 / 62 / 568 | 4,294,966,728 / 4,294,967,272 / 4,294,967,288 |
| `selfplay_v1` | 3,255 | 0 / 34 / 69 / 86 | 4,294,967,210 / 4,294,967,262 / 4,294,967,296 |
| `puzzle_v3` | 468 | 7 / 11 / 59 / 149 | 4,294,967,147 / 4,294,967,285 / 4,294,967,289 |
| forcing-19 | 19 | 9 / 41 / 103 / 149 | 4,294,967,147 / 4,294,967,255 / 4,294,967,287 |

**MEASURED.** The useful proof depths must therefore be short local clocks.
Neither a 4.295-billion-ply surrogate nor `∞` makes a short lower bound fire.

## 3. Candidate `N_h` families

### C1. Phase-aware census lower bound

**HYPOTHESIS — proof shape.** Let `c(P)` be the maximum attacker count in an
attacker-pure length-six window.  Define `N_h^census(P)` as
`censusLowerBound(P.phase,c(P)) ≤ h`.  The current exact table is
`FirstStone=[10,10,9,6,2,1]` and
`SecondStone=[12,12,9,5,4,1]`
([`tss_solver.rs`, lines 202–218](../packages/hexfield_eq/rust/src/tss_solver.rs#L202)).
The proof is the existing CensusBlocking induction: every attacker placement
can advance only the accounted alive-window census while the defender takes
the prescribed greedy blocker; phase determines when one versus two attacker
placements remain.  This definition is valid for every knob value `h`, but
“generalizing past eight” has no new numerical content: the largest table
entry is 12, so the condition becomes vacuous at `h≥12`.

### C2. Optimistic stone-deficit lower bound

**HYPOTHESIS — proof shape.** Let
`δ(P)=min({6-count_A(W) | W attacker-pure and touched} ∪ {6})`, and let
`A_phase(h)` be the maximum placements the current mover can receive in the
next `h` physical plies.  Define `N_h^deficit(P)` as
`δ(P) ≤ A_phase(h)`.  In the eventual winning window, every missing initial
attacker stone must be supplied by a distinct attacker placement.  A window
not currently touched starts at deficit six.  Ignoring every future defender
block only helps the attacker, so induction on the placement trace proves the
lower bound.  This is full-game sound but intentionally optimistic.

### C3. `NoJointCarrier` constant control

**HYPOTHESIS — proof-ready, restricted contract.** At a fresh attacker turn
with no own win-now, a forcing pair can activate a current count-three window
with one cell or a count-two window with two.  If no two distinct future
threats are jointly activatable, every pair has `τ≤1`, while
`vcf_pair_complete` admits only `τ≥2`.  Therefore the first contract turn is
impossible.  Define `N_h^joint(P)` to require a joint carrier for every `h`,
including `∞`.  This is the inherited `NoContractWin` theorem target, not a
global Connect6 loss theorem.

### C4. Exact `NoAdmissibleFirstTurn` control

**HYPOTHESIS — proof-ready, restricted contract.** Enumerate exact `T(P)` and
`S(P,a)`, then reject every pair that creates no attacker threat, leaves a
defender win-now unhit, or has `τ<2`.  If no pair remains, the restricted
choice node has no child at any `h`.  The enumeration mirrors
`WideTurnGate::second_candidates` and `evaluate_pair`
([`tss_solver.rs`, lines 9393–9444](../packages/hexfield_eq/rust/src/tss_solver.rs#L9393),
[`tss_solver.rs`, lines 9458–9550](../packages/hexfield_eq/rust/src/tss_solver.rs#L9458)).
This is an exact expansion certificate and useful as a control.  It is not a
defender-survivability schema, not absolute, and not a qualifying parametric
ladder family.

## 4. Numeric bite

### 4.1 Finite rungs

**MEASURED.** Hits are root dismissals under the candidate lower bound.  The
constant controls are omitted from this table because their count is unchanged
at every rung.

| cohort | census h2 / h4 / h8 / h16 | deficit h2 / h4 / h8 / h16 |
|---|---:|---:|
| grinds | 248 / 227 / 27 / 0 | 248 / 227 / 0 / 0 |
| human | 2,701 / 2,555 / 1,063 / 0 | 2,701 / 2,555 / 63 / 0 |
| self-play | 3,124 / 3,060 / 1,830 / 0 | 3,172 / 3,108 / 674 / 0 |
| puzzle | 463 / 452 / 200 / 0 | 463 / 452 / 4 / 0 |
| forcing-19 | 19 / 19 / 4 / 0 | 19 / 19 / 0 / 0 |

**MEASURED.** This is the entire numerical story of the finite ladder: strong
base cases, some census bite at eight, and zero everywhere by sixteen.  At the
sound `D=∞`, both finite lower bounds fire zero times.

### 4.2 Sound-deadline and infinity controls

The “at `D_i16`” column is diagnostic only; the true `D=∞` column is identical
for the constant controls and zero for both finite bounds.

| candidate / cohort | root hits at `D_i16` | fresh-turn rate | available Unknown wall dismissed |
|---|---:|---:|---:|
| census, every cohort | 0 | — | 0% |
| deficit, every cohort | 0 | — | 0% |
| `NoJointCarrier`, grinds | 0/248 | 0/193 = 0% | 0/599.491 s = 0% |
| `NoJointCarrier`, self-play | 482/3,255 | 482/1,610 = 29.94% | 0.102% of 1.473 s sample |
| `NoJointCarrier`, human | 182/2,720 | 182/1,322 = 13.77% | 0.107% of 0.875 s sample |
| `NoJointCarrier`, puzzle | 8/468 | 8/331 = 2.42% | 0.013% of 0.657 s sample |
| `NoJointCarrier`, forcing-19 | 0/19 | 0/18 = 0% | 0/11.805 s = 0% |
| exact no-turn, grinds | 0/248 | 0/193 = 0% | 0/599.491 s = 0% |
| exact no-turn, self-play | 873/3,255 | 873/1,610 = 54.22% | 0.456% of 1.473 s sample |
| exact no-turn, human | 684/2,720 | 684/1,322 = 51.74% | 0.879% of 0.875 s sample |
| exact no-turn, puzzle | 132/468 | 132/331 = 39.88% | 0.174% of 0.657 s sample |
| exact no-turn, forcing-19 | 2/19 | 2/18 = 11.11% | 0/11.805 s = 0% |

**MEASURED.** Exact no-turn beats the `NoJointCarrier` breadth percentages but
does not beat it on the intended prize: both dismiss zero grind roots and zero
grind wall.  Its larger breadth is dominated by positions the engine already
finishes at the first expansion.

## 5. False-dismissal battery

**MEASURED — PASS as a proxy.** Every candidate had **0 violations among
2,941 unique known-WIN roots**.  The union includes all 2,600 certified WIN
rows in the 47,902-row opening atlas, every `puzzle_v3` WIN row, every
forcing-corpus expected WIN, all 57 deep grind WINs, the Lane-C human/atlas/
forcing WINs, and current main4 dev WINs.  Overlaps are deduplicated by
position id.  The finite candidates were evaluated at the carrier surrogate
deadline when no certificate depth was recorded and at the exact certified
relative depth on 2,676 rows where one was recorded; the constant controls
were evaluated at `∞`.

**HYPOTHESIS — interpretation limit.** Zero empirical violations is not a
proof.  In particular, `NoJointCarrier` and exact no-turn prove only
`NoContractWin VcfPairComplete`; using either as a full-game LOSS would remain
unsound even though this battery happens to contain no counterexample.

## 6. Exact shallow ground truth

### 6.1 `h=2`: current-turn completion

**CODE-FACT.** The engine sets `own_win_now` exactly when the mover has a pure
count-five window, or a pure count-four window with two placements left
([`threats_shared.rs`, lines 157–179](../packages/hexo_models/rust/src/threats_shared.rs#L157)).
At fresh `FirstStone`, this is exactly a win in the mover's one two-stone turn;
at `SecondStone`, only count five qualifies.  Win is checked after each
placement, so no second stone is required after a winning first stone.

### 6.2 `h≤4`: standing double-threat loss

**CODE-FACT.** Let `F` be the opponent's current count-four/count-five threat
family and `b∈{1,2}` the mover's remaining placements.  The engine's exact
shallow forced-loss predicate is
`¬own_win_now ∧ τ(F)>b`
([`threats_shared.rs`, lines 57–76](../packages/hexo_models/rust/src/threats_shared.rs#L57)).
The certificate resolves in `b+2≤4` further placements
([`tss_solver.rs`, lines 2340–2349](../packages/hexfield_eq/rust/src/tss_solver.rs#L2340)).
At a defender `FirstStone` node, `τ(F)>2` is the familiar unanswerable
double-/multi-threat condition.

**MEASURED.** Exhaustive cohort counts were:

| cohort | exact current-turn win (`h=2` envelope) | exact standing-threat forced loss (`h≤4`) |
|---|---:|---:|
| grinds | 0 | 0 |
| human | 19 | 131 |
| self-play | 83 | 12 |
| puzzle | 5 | 10 |
| forcing-19 | 0 | 2 |

**MEASURED.** The independent Python predicates matched the engine diagnostic
on all 6,294 unique roots: 0 `own_win_now` mismatches and 0 `forced_loss`
mismatches.  These are exact shallow tactical base cases.  The `h≤4` result is
not an iff characterization of every arbitrary full-game construction within
four plies; it is the exact existing-threat/λ¹ verdict and a sound implication
to a win within four.

## 7. Verdict and reduced Lean program

**MEASURED — bar evaluation.** No candidate reaches the first GO arm:
sound-deadline grind Unknown-wall dismissal is 0%, below 10%.  No new
schema-survivability `h=∞` member reaches the second arm.  The only predicate
broader than `NoJointCarrier` is exact no-turn, an already-known exhaustive
contract expansion rather than a maintainable-forever invariant.  It yields
0% grind wall and therefore does not license the parametric program.

**HYPOTHESIS — program size.** The reduced program should contain three
endpoints and roughly 12–18 supporting lemmas: phase/placement arithmetic and
window completion; rank-≤2 hitting facts and the `b+2` bridge; then legality,
residual-`τ=1`, commutation, and cardinality for boundary pairs.  It should not
define or induct over a general `N_h` schema in this phase.

The following theorem shapes are ready for a Lean session; names of existing
project primitives may be substituted, but the quantifiers and scopes should
not be weakened.

```lean
def OwnWinThisTurn (P : Position) (A : Player) : Prop :=
  ∃ W : Window,
    count A.other W P = 0 ∧
    (count A W P = 5 ∨
      (P.phase = .firstStone ∧ count A W P = 4))

theorem attackerWinsWithinTwo_iff_ownWinThisTurn
    (hphase : P.phase = .firstStone)
    (hmover : P.toMove = A)
    (hnt : Nonterminal P) :
    AttackerWinsWithin P A 2 ↔ OwnWinThisTurn P A

def StandingThreatFamily (P : Position) (A : Player) :
    Finset (Finset Cell) :=
  liveThreatWindows P A |>.image (fun W => emptyCells W P)

theorem lambdaOneForcedLoss_iff_hittingNumber_exceeds_budget
    (hnt : Nonterminal P) :
    LambdaOneForcedLoss P ↔
      ¬ OwnWinThisTurn P P.toMove ∧
      turnBudget P < hittingNumber (StandingThreatFamily P P.toMove.other)

theorem opponentWinsWithinFour_of_lambdaOneForcedLoss
    (h : LambdaOneForcedLoss P) :
    AttackerWinsWithin P P.toMove.other 4

def MinimumCoverPairs (F : Finset (Finset Cell)) :
    Finset (Sym2 Cell) :=
  {p | p.1 ≠ p.2 ∧ ∀ E ∈ F, p.1 ∈ E ∨ p.2 ∈ E}

theorem forcedB2PairQuotient
    (hpost : P.phase = .firstStone)
    (hdef : P.toMove = A.other)
    (hnt : Nonterminal P)
    (hnow : ¬ OwnWinThisTurn P A.other)
    (hrank : ∀ E ∈ StandingThreatFamily P A, E.card ≤ 2)
    (htau : hittingNumber (StandingThreatFamily P A) = 2) :
    let M := MinimumCoverPairs (StandingThreatFamily P A)
    M.card ≤ 4 ∧
    (∀ p ∈ M,
      LegalOrderedPair P p.1 p.2 ∧
      LegalOrderedPair P p.2 p.1 ∧
      playPair P p.1 p.2 = playPair P p.2 p.1) ∧
    ExactNonLosingDefenderPairs P A M
```

**HYPOTHESIS — important statement boundary.** The first theorem is explicitly
fresh-turn because two physical placements then belong to the attacker.  Add a
separate `SecondStone` theorem (`count=5`) if that phase is needed; do not hide
the phase distinction in a loose clock convention.

## 8. Reproduction

```powershell
python .scratch\deadline_ladder_r.py

# Optional compiled-engine equality check (after building the two existing
# Python extensions under the brief's target/CARGO_TARGET_DIR constraints):
python .scratch\deadline_ladder_r.py --engine-check `
  --package-root .scratch\nativecheck
```

**MEASURED.** No engine, verifier, Lean, test, or existing tracked source file
was edited.  The only deliverables are this report and `.scratch` artifacts.
