# Horizon extension R2: exact fresh-turn decisions through eight plies

## Executive verdict

**MEASURED.** The finite-horizon program reaches `h<=8` exactly.  The scratch
decider searches the full Connect6 game, not the forcing contract, over a
proved-relevant finite quotient.  It uses no board-radius, node, wall, or
candidate cap.  On 3,474 real fresh-turn roots, all internal nesting checks
passed, and all 76 unique available engine-certified WIN roots with certificate
depth at most eight were caught at both requested horizons: **zero misses**.

**CODE-FACT.** At a fresh `FirstStone` turn the physical-player schedule is

`A,A,D,D,A,A,D,D`.

Consequently the current attacker has no placement after ply 6.  The exact
attacker predicates at `h<=6` and `h<=8` are therefore identical.  The loss
duals are not identical: the defender gets a second pair at plies 7--8.

**MEASURED.** Exact current-attacker `WinWithin6 = WinWithin8` fires on
54/1,610 self-play roots (3.35%), 94/1,322 human roots (7.11%), 14/331 puzzle
roots (4.23%), and zero of 193 fresh grind or 18 fresh forcing-19 roots.  Exact
`ForcedLossWithin8` fires on 8 (0.50%), 101 (7.64%), 10 (3.02%), 0, and 2
(11.11%) respectively.

**HYPOTHESIS -- Lean status.** The relevance and endpoint proofs below are
complete proof sketches and the executable predicates are exhaustive, but the
statements have not yet been accepted by Lean.  “Exact” in this report means
exactly enumerated against the stated game semantics and relevance theorem; it
does not claim completed formalization.

## 1. Scope, artifacts, and clocks

**MEASURED.** The implementation is
[`horizon_r2.py`](../.scratch/horizon_r2.py), its complete cohort output is
[`horizon_r2.json`](../.scratch/horizon_r2.json), and the independent battery is
[`horizon_r2_validate.py`](../.scratch/horizon_r2_validate.py) with output
[`horizon_r2_validation.json`](../.scratch/horizon_r2_validation.json).  It
inherits only the Phase-R move replay, phase arithmetic, and length-six window
reconstruction.  No engine source, verifier source, or tracked file other than
this report was edited; no cargo build was used.

**CODE-FACT.** A horizon counts physical placements and a win terminates at its
first winning prefix.  This report's principal scope is a nonterminal fresh
turn (`P.phase = FirstStone`, `P.toMove = A`).  The generic validation decider
also handles `SecondStone` roots.  The requested schedules and per-player
placement quotas are:

| horizon | schedule from a fresh turn | `k_A` | `k_D` |
|---:|---|---:|---:|
| 6 | `A,A,D,D,A,A` | 4 | 2 |
| 8 | `A,A,D,D,A,A,D,D` | 4 | 4 |

For deciding an `A` win, trailing moves after A's last placement are
irrelevant.  Thus the `h=8` attacker endpoint safely uses the smaller `h=6`
universe (`k_A=4,k_D=2`).  The `h=8` loss dual needs the full `4,4` universe.

## 2. Load-bearing relevance lemmas

Let `c_p(W,P)` be player `p`'s stone count in length-six window `W` at the
root.  Let `E(W,P)` be its root-empty cells.  For a schedule `s`, let `k_p(s)`
be the number of placements assigned to `p`.  Define

`C_p(P,s) = { W | c_(1-p)(W,P)=0 and |E(W,P)| <= k_p(s) }`

and the finite relevant universe

`U(P,s) = union { E(W,P) | p in {0,1}, W in C_p(P,s) }`.

### 2.1 Root-window ancestry

**HYPOTHESIS -- relevance lemma.** For every legal trace `sigma` of length at
most `|s|`, every player `p`, and every window `W` that is first completed by
`p` along `sigma`, `W` belongs to `C_p(P,s)`.  Moreover, every cell of `W`
played after the root lies in `U(P,s)`.

**Proof sketch.** Stones are permanent.  A finally `p`-pure window contained no
opponent stone at the root.  At most `k_p(s)` new `p` stones occur before the
deadline, so the root had at least `6-k_p(s)` `p` stones in `W`, equivalently
`|E(W,P)|<=k_p(s)`.  Every newly filled cell of `W` was root-empty, hence is in
the union defining `U`.  This quantifies over both players and every terminal
prefix; it does not assume forcing play.

The condition `k_p(s)<6` is what makes the union finite on `Z^2`: every
qualifying window contains at least one root stone, and a finite position
touches only finitely many length-six windows.  At `h=6`, qualifying attacker
windows have at least two root attacker stones and qualifying defender windows
have at least four.  At `h=8`, both sides have at least two.

### 2.2 Outside moves and the inert class

**HYPOTHESIS -- quotient lemma.** If `x` is root-empty and `x notin U(P,s)`,
then placing at `x` cannot contribute to or block any win completed by the
deadline.  Projecting every such placement to an inert action preserves every
terminal prefix through `s`.

**Proof sketch.** If `x` contributed to a completed mover window, or blocked a
window the opponent otherwise completed, that completed window would satisfy
root-window ancestry and its root-empty cell `x` would belong to `U`, a
contradiction.  Induction over the trace therefore preserves all relevant
window occupancies.  There are always enough concrete inert representatives:
the root and `U` are finite while the rules board is `Z^2`, so choose fresh
cells more than five steps from them and one another.

**HYPOTHESIS -- dominance corollary.** When an empty relevant cell exists, an
inert placement is weakly dominated for the mover by a relevant placement.
Adding one's stone early can only advance one's live windows and destroy the
opponent's; it cannot destroy one's own window or create an opponent window.
An induction that substitutes a later intended use of that cell by the inert
representative proves the strategy statement.  The decider therefore keeps
one inert class only when no relevant placement remains.

This last corollary is why the implementation's finite action set is exact,
not merely sound for positive answers.  A claim that every literal move of an
arbitrary winning trace lies in `U` would be false because a winning trace may
contain wasted remote placements; the normalization theorem is the necessary
and sufficient statement.

### 2.3 Measured universe sizes

**MEASURED.** Sizes below are `p50 / p90 / max` over fresh roots.  `U6` is used
by the attacker endpoints and the `h=6` dual; `U8` is used by the `h=8` dual.

| cohort | roots | `U6` | `U8` |
|---|---:|---:|---:|
| self-play | 1,610 | 20 / 47 / 77 | 46 / 88 / 130 |
| human | 1,322 | 23 / 47 / 165 | 48 / 83 / 260 |
| puzzle | 331 | 20 / 44 / 81 | 41 / 80 / 151 |
| grinds | 193 | 38 / 58 / 77 | 74 / 107 / 130 |
| forcing-19 | 18 | 43 / 53 / 75 | 74 / 91 / 134 |

## 3. Exact decision procedures

Write `tau(F)>2` when a finite family of nonempty cell sets has no hitting set
of size at most two.  After a pair, every live next-turn win window has one or
two empty cells, so this hitting test is complete.

### 3.1 Current attacker at h=6 and h=8

**HYPOTHESIS -- exact characterization.** Enumerate every normalized unordered
attacker pair `a` in `U6` (with winning first-placement prefixes retained).
Let `F_A(P+a)` be the residual-empty family of all A-pure windows with at most
two empties.  Then

`AttackerWinsWithin(P,A,6)` iff some `a` either completes six immediately, or

1. leaves no D-pure window completable by D's intervening pair, and
2. satisfies `tau(F_A(P+a))>2`.

**Proof sketch.** In the only nonterminal continuation, D has exactly two
placements before A's final pair.  D prevents every A completion exactly when
those two placements hit every member of `F_A`; unused capacity is filled by
an inert move.  If D instead has a current-turn completion, D wins first.  If
neither defense exists, at least one A window remains and A fills its at-most
two residual cells.  Conversely either defense refutes that root pair.  Pair
enumeration plus the relevance quotient covers every attacker strategy.

**CODE-FACT.** A has no placement at plies 7--8, hence

`AttackerWinsWithin(P,A,8) <-> AttackerWinsWithin(P,A,6)`

at every fresh A turn.  The code deliberately evaluates both requested entry
points and their winning-ID sets are exactly equal in every cohort.

### 3.2 Forced-loss duals

**HYPOTHESIS -- h=6 dual.** D's last placement before the deadline is ply 4.
Thus D forces a win within six iff, for every normalized initial A pair that
does not already complete A's six, some D-pure window remains with at most two
empties.  D then fills it on its only pair.  This is also the exact h=4 dual;
plies 5--6 cannot create a D win.

**HYPOTHESIS -- h=8 dual.** For every normalized initial A pair `a` that does
not win, there must exist a normalized D pair `d` such that either D wins on
`d`, or both:

1. `d` hits every A-pure window that A could complete on plies 5--6; and
2. the residual D threat family has `tau>2`.

Then every intervening A pair leaves a D window, which D fills at plies 7--8.
If either clause fails, A chooses the corresponding winning or covering pair,
so D has no forcing strategy.  This is a complete `forall A, exists D`
enumeration, not a forcing-class approximation.

### 3.3 Generic implementation

**CODE-FACT.** The independent generic decider represents the cells of `U` as
bit positions and root-completable windows as bit masks.  At each physical
placement it evaluates all normalized relevant actions, checks both players'
terminal windows, prunes only when the target lacks enough scheduled
placements to complete any live root window, and memoizes
`(schedule index,P0 mask,P1 mask)`.  The fresh-turn implementation uses the
closed pair/hitting-set characterizations above.  Forty general real roots and
12 shallow certified WIN roots were evaluated by both implementations with
zero verdict mismatches.  The generic up-to-h entry point runs exact smaller
clocks first and uses horizon monotonicity to short-circuit a witnessed WIN;
if none fires, it exhausts the requested clock.

## 4. Complexity and firing rates

### 4.1 Exhaustion cost

**MEASURED.** “Nodes” are normalized candidate pairs inspected (including
nested D replies for the h=8 dual).  Times are single-process CPython 3.14 wall
on the five complete fresh-root cohorts; there was no warm cache shared between
positions.  This is production-shaped position data, but not production Rust
latency.

| cohort | current h6/h8 nodes p50 / p90 / max | current mean ms | loss h6 mean ms | loss h8 nodes p50 / p90 / max | loss h8 mean / max ms |
|---|---:|---:|---:|---:|---:|
| self-play | 171 / 1,035 / 2,926 | 3.83 | 2.57 | 596 / 4,372 / 1,345,240 | 37.35 / 8,565.32 |
| human | 210 / 990 / 9,730 | 4.45 | 2.95 | 904 / 18,590 / 6,042,901 | 113.54 / 32,628.93 |
| puzzle | 190 / 903 / 2,926 | 2.86 | 2.53 | 1,100 / 7,377 / 1,140,310 | 55.20 / 4,495.14 |
| grinds | 703 / 1,653 / 2,926 | 6.67 | 4.01 | 2,146 / 5,357 / 39,162 | 27.00 / 149.86 |
| forcing-19 | 903 / 1,378 / 2,775 | 6.37 | 6.04 | 2,017 / 8,002 / 702,598 | 250.32 / 2,239.86 |

The h=8 dual is exact but has a severe true/universal tail.  Across all 3,474
roots the complete regenerated measurement took 287.6 seconds; the simpler
attacker endpoint is consistently millisecond-scale in Python.

### 4.2 Firing rates by rung

**MEASURED.** Denominators are fresh `FirstStone` roots, not all rows.  For the
current attacker, h=4 equals h=2 because the intervening D pair cannot create
an A win.  For the loss dual, h=6 equals h=4 because D has no new placement.

| cohort | current win h2 / h4 / h6 / h8 | forced loss h2 / h4 / h6 / h8 |
|---|---:|---:|
| self-play (1,610) | 43 / 43 / 54 / 54 (2.67 / 2.67 / 3.35 / 3.35%) | 0 / 6 / 6 / 8 (0 / 0.37 / 0.37 / 0.50%) |
| human (1,322) | 19 / 19 / 94 / 94 (1.44 / 1.44 / 7.11 / 7.11%) | 0 / 67 / 67 / 101 (0 / 5.07 / 5.07 / 7.64%) |
| puzzle (331) | 5 / 5 / 14 / 14 (1.51 / 1.51 / 4.23 / 4.23%) | 0 / 5 / 5 / 10 (0 / 1.51 / 1.51 / 3.02%) |
| grinds (193) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| forcing-19 (18) | 0 / 0 / 0 / 0 | 0 / 2 / 2 / 2 (0 / 11.11 / 11.11 / 11.11%) |

## 5. Validation battery

### 5.1 Internal implications

**MEASURED -- PASS.** Over all 3,474 fresh roots, each cohort separately had
zero violations of:

- Phase-R exact current-turn win implies `WinWithin6`;
- `WinWithin6` implies `WinWithin8` (in fact equality held);
- Phase-R standing-threat forced loss implies `ForcedLossWithin6`; and
- `ForcedLossWithin6` implies `ForcedLossWithin8`.

The base predicates are the independently reconstructed Phase-R predicates
that previously matched the engine threat diagnostic on 6,294 unique roots.

### 5.2 Engine certificate ground truth

**MEASURED -- PASS.** The inherited known-WIN registry reproduced exactly
2,941 unique roots: 2,676 have an exact certificate depth and 265 are undated.
There are 76 unique roots with depth at most eight (77 raw cohort/atlas
references before deduplication).  All 76 were caught by the h=6 decider and
all 76 by h=8: zero misses.  Partial-turn certificates were decided at their
requested h=6/h=8 endpoint by the generic exact monotone ladder.  No available
shallow certificate had depth seven or eight; depths one, two, five, and six
were represented.

**MEASURED -- adapted 2,941 battery.** Of the registry, 2,600 depth-stamped
wins resolve later than eight and 265 lack an exact depth.  They are not valid
counterexamples to `NoWinWithin8`: a bounded partial refutation is compatible
with a later full-game win.  The adapted false-dismissal battery therefore uses
all 2,941 to audit classification and uses exactly the 76 depth-eligible roots
as ground truth.  It has zero false dismissals.  No `NoWinWithin8` result is
relabeled a full-game LOSS.

### 5.3 Engine Unknown and width witnesses

**MEASURED.** Seventeen exact h=8 wins have a corresponding available engine
record (2 self-play, 14 human, 1 puzzle); all 17 engine records are already
WIN.  There are **zero** exact h=8 wins among the available engine-Unknown
join, hence no new shallow forcing-width witness.

**MEASURED.** The five known atlas width-exhaust rows were checked explicitly
and all five are `NoWinWithin8`.  In particular the three J2near witnesses
`oa-0153903c5a863630`, `oa-6fda812864c6d19a`, and
`oa-773ca1a59e95f4e1` are negative here, as they must be: their certified
terminal lines require 22, 21, and 22 placements.  R2 therefore neither
rediscovers nor contradicts the deep free-tempo width mechanism.

## 6. Bite and honest consumption paths

**HYPOTHESIS -- MCTS leaf verdicts.** The h=6 attacker predicate has useful
root bite (3.35% self-play, 7.11% human) and millisecond Python cost.  A compiled
bitset version could be evaluated as a leaf fast-WIN, returning the witnessing
first pair.  The present Python numbers do not justify a production hot-path
claim.  The h=8 loss dual's 8.6--32.6 second observed tail rules out direct
leaf use in this form.

**HYPOTHESIS -- atlas and labeling.** `WinWithin{2,6}` and
`ForcedLossWithin{4,8}` are exact shallow tiers suitable for deterministic
labels.  Their clock equalities should be stored explicitly rather than
presenting four independent information levels.

**HYPOTHESIS -- solver seed and ordering.** A positive attacker result supplies
an exact winning first pair; a positive dual supplies exact defender response
pairs.  These can seed legal move ordering without pruning wider search.

**HYPOTHESIS -- certified refutation leaves.** A negative result is the
checkable statement `NoWinWithin8`, not LOSS.  A certificate needs the root
window census, relevance-universe digest, exhaustive normalized pair list, and
per-pair two-cover or opponent-completion witness.  This is a sound base leaf
for a bounded refutation grammar and a useful “not shallow” label.

## 7. Lean-ready statements

**HYPOTHESIS.** Names may be adapted to the concurrent development, but the
players, windows, traces, terminal prefixes, and fresh-phase quantifiers below
must not be weakened.

```lean
def RootCompletableWindows
    (P : Position) (p : Player) (s : List Player) : Finset Window :=
  touchedWindows P |>.filter (fun W =>
    count p.other W P = 0 ∧ emptyCount W P ≤ s.count p)

def RelevantCells (P : Position) (s : List Player) : Finset Cell :=
  Finset.univ.biUnion (fun p =>
    (RootCompletableWindows P p s).biUnion (fun W => emptyCells W P))

theorem firstWinningWindow_rootCompletable
    (hlegal : LegalTraceFrom P σ)
    (hlen : σ.length ≤ s.length)
    (hschedule : Owners σ = s.take σ.length)
    (hfinite : ∀ q, s.count q < 6)
    (hfirst : FirstWinAt P σ i p W) :
    W ∈ RootCompletableWindows P p s ∧
    ∀ j < i, cellAt σ j ∈ W → cellAt σ j ∈ RelevantCells P s

theorem outsideRelevant_outcomeInert
    (hfinite : ∀ p, s.count p < 6)
    (hx : x ∉ RelevantCells P s)
    (hempty : Empty P x) :
    DeadlineOutcomeProjection P s (play P p x) =
      DeadlineOutcomeProjection P s (inertPlay P p)

theorem finiteHorizon_relevance_iff
    (hnt : Nonterminal P)
    (hfinite : ∀ p, s.count p < 6) :
    AttackerWinsOnSchedule P A s ↔
      AttackerWinsRestricted P A s (RelevantCells P s ∪ {inertCell P s})

def Fresh6 (A : Player) : List Player := [A,A,A.other,A.other,A,A]
def Fresh8 (A : Player) : List Player :=
  [A,A,A.other,A.other,A,A,A.other,A.other]

def PairFork6 (P : Position) (A : Player) (a : Sym2 Cell) : Prop :=
  LegalNormalizedPair P A a ∧
  (CompletesPair P A a ∨
    (¬ CanCompleteThisTurn (playPair P a) A.other ∧
     2 < hittingNumber (NextTurnThreatFamily (playPair P a) A)))

theorem attackerWinsWithinSix_iff_pairFork
    (hphase : P.phase = .firstStone)
    (hmover : P.toMove = A)
    (hnt : Nonterminal P) :
    AttackerWinsWithin P A 6 ↔ ∃ a, PairFork6 P A a

theorem attackerWinsWithinEight_iff_withinSix
    (hphase : P.phase = .firstStone)
    (hmover : P.toMove = A)
    (hnt : Nonterminal P) :
    AttackerWinsWithin P A 8 ↔ AttackerWinsWithin P A 6

def DefenderWinsByFourAfter (P : Position) (A : Player)
    (a : Sym2 Cell) : Prop :=
  ¬ CompletesPair P A a ∧
  CanCompleteThisTurn (playPair P a) A.other

theorem opponentWinsWithinSix_iff_allFirstPairs
    (hphase : P.phase = .firstStone)
    (hmover : P.toMove = A)
    (hnt : Nonterminal P) :
    AttackerWinsWithin P A.other 6 ↔
      ∀ a, LegalNormalizedPair P A a → DefenderWinsByFourAfter P A a

def DefenderReplyFork8 (P : Position) (A : Player)
    (a d : Sym2 Cell) : Prop :=
  LegalNormalizedPair P A a ∧ ¬ CompletesPair P A a ∧
  LegalNormalizedPair (playPair P a) A.other d ∧
  (CompletesPair (playPair P a) A.other d ∨
    (¬ CanCompleteThisTurn (playPair (playPair P a) d) A ∧
     2 < hittingNumber
       (NextTurnThreatFamily (playPair (playPair P a) d) A.other)))

theorem opponentWinsWithinEight_iff_replyFork
    (hphase : P.phase = .firstStone)
    (hmover : P.toMove = A)
    (hnt : Nonterminal P) :
    AttackerWinsWithin P A.other 8 ↔
      ∀ a, LegalNormalizedPair P A a →
        ¬ CompletesPair P A a ∧ ∃ d, DefenderReplyFork8 P A a d
```

The implementation's unordered pair quotient retains terminal one-stone
prefixes.  A Lean `LegalNormalizedPair` must do the same; it cannot silently
require a second placement after a first-placement win.

## 8. Where the ladder actually stops

**CODE-FACT -- frontier.** The present relevance theorem stops at `h=8`.  At a
fresh `h=10` clock the attacker receives six placements
(`A,A,D,D,A,A,D,D,A,A`), so a winning window may contain zero attacker stones
at the root.  There are then infinitely many translated root-empty candidate
windows on `Z^2`, and `U(P,s)` is no longer finite.  `h=12` gives the defender
six placements as well, so the dual loses the same anchor.  A finite-radius or
“touched windows only” implementation at either rung would be one-sided and is
not reported as exact.  Reaching h=10 would require a new theorem quotienting
remote empty-board components by translation and finite trace shape; reaching
h=12 would require that theorem symmetrically for both players.  Without it,
the exhaustive move universe and hence the node cost are undefined/infinite,
not a responsible extrapolation of the h=8 timings.  This is the measured
program frontier: exact through eight, theorem-blocked at ten.

## 9. Reproduction and hashes

```powershell
python .scratch\horizon_r2.py --fresh-only --out .scratch\horizon_r2.json
python .scratch\horizon_r2_validate.py
```

**MEASURED.** Run at commit `72f68ced1b01e3e97a863eef2a37fc635d6ed74e`
with Python 3.14.0.  SHA-256:

- `horizon_r2.py`: `76A9F5B10939D9810A15E9539D4B4FEE13A3E70179E2914BCC8D0A8E16544257`
- `horizon_r2.json`: `EB70D62FD47B02A120C7ED6823143B3A1981034A117089986B94D1417245CC67`
- `horizon_r2_validate.py`: `5CF83CE04BB4E51E695F11A2159321ACA0B9B30DC83095B629B660C536CA7521`
- `horizon_r2_validation.json`: `4509962A94117B8860CB31C1CDF2AB5AAA80A18F32D90DF27D8590692ECBB19B`
