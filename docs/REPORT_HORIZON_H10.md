# Horizon H10: translation quotient, production bite, and H8 port handoff

## Executive verdict

**HYPOTHESIS -- proof-ready.** The h=10 infinity obstruction closes.  Let U be
the finite root-anchored h10 universe and add every root-empty length-six
window meeting U.  Every remaining translated empty-board component collapses
to one constant subgame, and that constant is LOSS for the attacker: after the
first four attacker placements, all candidate six-windows have a two-cell
cover.  The resulting exact endpoint is a finite `exists A-pair, forall
D-pair, exists A-pair` search with D's final pair discharged by a two-cover
test.  No radius or coordinate-carrier bound is used.

**MEASURED.** The constant check exhausts all ten normalized four-stone line
shapes.  Nine have a one-cell cover; the consecutive-four shape has the
two-cover `{-2,4}`.  All ten pass.

**MEASURED -- positive validation.** The known-WIN registry has 123 unique
roots with certificate depth at most ten, up from 76 at depth at most eight.
The 76 old roots are caught by the exact h8 shortcut.  Two newly eligible
depth-10 fresh roots were completed by the new quotient with witnesses:

| root | nodes | wall | first pair |
|---|---:|---:|---|
| `human_b132a09ccb4eb829_p101` | 34,703 | 1.176 s | `(-4,-5),(12,-12)` |
| `sp_20_p77` | 689,531 | 9.779 s | `(6,-12),(10,-19)` |

They establish genuine h10 bite.  The full 123-root plus atlas validation run
did not complete in a 1,200-second single-process frame, so 45 new eligible
certificates were not evaluated by this Python implementation.  This report
does not turn a timeout into a miss or a negative verdict.

**MEASURED -- complete production H8.** All 6,443 frozen rows and all 248 grind
rows were evaluated at h<=8, including `SecondStone` roots.  Current-attacker
WIN / forced-LOSS totals are 101/17 self-play, 157/201 human, 20/22 puzzle,
and 0/0 grinds.  The partial-turn specialization agrees with independent
generic minimax on 20 roots with zero mismatches.

**MEASURED -- H10 evidence boundary.** Internal `WinWithin8 => WinWithin10`
was audited on all 3,474 fresh roots with zero violations.  Exhaustive h10
firing rates and the five h10 atlas negatives did not complete: representative
negative cohort runs repeatedly entered multi-minute universal tails.  The
reported cohort h10 numbers are therefore certified floors, not firing rates.

## 1. Artifacts and semantics

The new decider and evidence are
[`horizon_h10.py`](../.scratch/horizon_h10.py) and
[`horizon_h10.json`](../.scratch/horizon_h10.json).  Complete all-row h8
measurement is in
[`horizon_production_h8.py`](../.scratch/horizon_production_h8.py) and
[`horizon_production_h8.json`](../.scratch/horizon_production_h8.json).  The
Rust handoff is [`PORT_SPEC_HORIZON_H8.md`](PORT_SPEC_HORIZON_H8.md).

**CODE-FACT.** A fresh h10 clock is

`A,A,D,D,A,A,D,D,A,A`,

so `k_A=6` and `k_D=4`.  A win terminates on its first winning placement.
At a `SecondStone` h10 root the current attacker receives only five placements,
so R2's original finite relevance theorem still applies; the translation
quotient is required only at fresh `FirstStone` roots.

## 2. The h10 translation quotient

### 2.1 Anchored core and near-empty halo

For player p, define the root-anchored completable family

```text
C+_p(P,s) = { W |
  W contains no root opponent stone,
  W contains at least one root p stone, and
  |E(W,P)| <= k_p(s) }.
```

Let

```text
U+(P,s) = union { E(W,P) | p in {A,D}, W in C+_p(P,s) }
N(P,s)  = { W | W is root-empty and W intersects U+(P,s) }
V(P,s)  = U+(P,s) union union N(P,s).
```

**HYPOTHESIS -- finiteness lemma.** `V(P,s)` is finite.  `C+` is finite by
R2 root-window ancestry because every member contains a root stone.  Each cell
of U lies in exactly 18 length-six windows, hence `|N| <= 18|U+|` and
`|V| <= |U+|+6|N|`.

### 2.2 The remote empty-board constant

**HYPOTHESIS -- four-stone two-cover lemma.** Let X be four distinct cells.
Let F(X) contain the residual two-cells of every length-six window containing
X.  Then F is empty or has a hitting set of size at most two.

**Proof sketch.** A window containing four stones puts all four on one of the
three axes with span at most five.  Translate the minimum coordinate to zero
and quotient reflection.  It is enough to enumerate the ten subsets
`{0} union choose(three,{1,...,5})`.  The executable enumeration lists every
length-six interval containing the subset and constructs a cover.  The only
rank-two shape is `{0,1,2,3}`, whose residuals are
`{-2,-1}`, `{-1,4}`, and `{4,5}`, covered by `{-1,4}` (the implementation may
choose the equivalent `{-2,4}`).  Every other shape has a common residual
cell.  D6 maps axes and reflections bijectively, so the axial proof covers all
board orientations.

**HYPOTHESIS -- remote constant theorem.** On an empty-enough region, A cannot
force six-in-a-row in six placements against the four intervening D placements.

**Proof sketch.** Immediately before A's final pair, A has placed exactly four
stones.  Every root-empty window A can finish contains all four.  D's second
pair plays the cover from the lemma.  D's first pair is not needed for this
remote defense.  Thus the requested position-independent constant is `false`.

### 2.3 Interaction normalization

**HYPOTHESIS -- h10 translation-quotient theorem.** For a nonterminal fresh
root P,

```text
WinWithin10(P,A)
  iff WinWithin8(P,A)
   or WinWithin10Restricted(P,A,V(P,s10)).
```

**Proof sketch.** Consider the branch after A's first pair.

- If a possible root-empty terminal window meets U+, it belongs to N and every
  one of its future cells belongs to V.
- If it does not meet U+, all six A placements on that terminal branch are
  outside U+.  Because the first pair changes no anchored window, the
  h8-negative defender response covers every anchored completion available to
  A's second pair (or wins first).  A's second pair also changes no anchored
  window.  D reserves the second pair for the remote two-cover lemma.  Hence
  the branch loses.
- An outside first-pair cell which is neither anchored nor shares a retained
  root-empty window with its mate cannot occur in any six-placement terminal
  window: such a window must contain both first-pair stones.  It is inert and
  is dominated by a relevant placement.  The same monotone substitution
  applies at later nodes after filtering windows by remaining placement quota.

This is the interaction term missing at R2.  D is allowed to split its first
pair arbitrarily across anchored defense, its own anchored construction, and
the retained near-empty windows.  Only the wholly remote class is normalized
away, and its final-pair defense is proved rather than assumed.

### 2.4 Exact finite endpoint

After a normalized first A pair `a`, enumerate every normalized D pair `d` in
the live finite union.  After every nonwinning `d`, A must have a pair `b` such
that it wins immediately or:

1. D has no pure window completable on its final pair; and
2. A's live residual family of size one or two has no two-cover.

Thus the endpoint is exactly

```text
exists a, forall d,
  not DCompletes(d) and
  exists b, ACompletes(b) or (not DCanCompletePair and tau(F_A)>2).
```

**CODE-FACT.** `.scratch/horizon_h10.py` implements this formula with arbitrary
precision integer bitsets, lazy exhaustive pair streams, first-placement
terminal prefixes, and no node/wall/radius cap.  Its completed boolean answers
are exact relative to the quotient hypotheses.  Runtime timeout is a harness
boundary, not part of the decider.

## 3. Validation and new bite

### 3.1 Nesting

**MEASURED -- PASS.** Across all 1,610 self-play, 1,322 human, 331 puzzle, 193
grind, and 18 forcing fresh roots, there were zero `h8=true,h10=false`
violations.  The h10 entry point intentionally evaluates the exact h8 decider
first; all 162 h8 positives were replayed through that path.  H8-negative roots
cannot violate the implication.

### 3.2 Certified ground truth

**MEASURED.** Registry accounting is 2,941 unique known wins, 123 eligible by
depth ten: 76 at depth at most eight, 4 at depth nine, and 43 at depth ten.
The old 76 are caught.  The two completed new tests above are caught with
explicit first pairs.  The attempted complete run exceeded 1,200 seconds, so
the honest result is `78 tested / 78 caught / 45 new untested`, not
`123/123`.

### 3.3 Cohort floors

The known depth-9/10 certificates give strict lower bounds on the new rung:

| cohort | roots | exact h8 wins | certified h10 floor | delta floor |
|---|---:|---:|---:|---:|
| self-play | 1,610 | 54 (3.354%) | >=55 (3.416%) | >=1 |
| human | 1,322 | 94 (7.110%) | >=99 (7.489%) | >=5 |
| puzzle | 331 | 14 (4.230%) | >=15 (4.532%) | >=1 |
| grinds | 193 | 0 | >=0 | >=0 |
| forcing-19 | 18 | 0 | >=0 | >=0 |

**MEASURED -- evidence limit.** These are not exhaustive h10 firing rates.
The human depth-10 root also appears in puzzle, so rows are cohort counts, not
a deduplicated registry count.  Even a three-row evenly spaced sample entered a
greater-than-five-minute negative; a five-prefix cross-cohort run did likewise.
No incomplete sample is reported as a rate.

### 3.4 Width-exhaust and J2near

**MEASURED -- incomplete at h10.** R2's exact h8 evaluation remains negative on
all five atlas width-exhaust rows, including the three J2near IDs
`oa-0153903c5a863630`, `oa-6fda812864c6d19a`, and
`oa-773ca1a59e95f4e1`.  Their known terminal lines require 22, 21, and 22
placements.  The requested h10 negative evaluations entered the same universal
tail and did not complete, so this lane neither upgrades those rows to
`NoWinWithin10` nor finds a contradiction.

## 4. Production h<=8 bite on all phases

The three frozen sets contain 6,443 rows.  Grinds are the requested 248-row
subset of self-play and are reported separately.  `opening` is retained as a
third phase rather than silently folded into `SecondStone`.

### 4.1 Firing rates

| cohort / phase | n | current WinWithin8 | ForcedLossWithin8 |
|---|---:|---:|---:|
| self-play / opening | 48 | 0 | 0 |
| self-play / FirstStone | 1,610 | 54 (3.35%) | 8 (0.50%) |
| self-play / SecondStone | 1,597 | 47 (2.94%) | 9 (0.56%) |
| **self-play / all** | **3,255** | **101 (3.10%)** | **17 (0.52%)** |
| human / FirstStone | 1,322 | 94 (7.11%) | 101 (7.64%) |
| human / SecondStone | 1,398 | 63 (4.51%) | 100 (7.15%) |
| **human / all** | **2,720** | **157 (5.77%)** | **201 (7.39%)** |
| puzzle / FirstStone | 331 | 14 (4.23%) | 10 (3.02%) |
| puzzle / SecondStone | 137 | 6 (4.38%) | 12 (8.76%) |
| **puzzle / all** | **468** | **20 (4.27%)** | **22 (4.70%)** |
| grinds / FirstStone | 193 | 0 | 0 |
| grinds / SecondStone | 55 | 0 | 0 |
| **grinds / all** | **248** | **0** | **0** |

**MEASURED.** Mid-turn leaves add 47 self-play, 63 human, and 6 puzzle exact
wins, plus 9, 100, and 12 exact losses.  This confirms production bite beyond
fresh roots, especially the human loss tier.  It adds no bite on the grind
target.

### 4.2 Cost and implementation note

**MEASURED.** All-row mean current/loss times were 4.52/21.53 ms self-play,
5.60/55.05 ms human, 3.66/44.64 ms puzzle, and 9.66/30.85 ms grinds.  Fresh
loss maxima remained the expensive tail: 6.94 s self-play, 33.62 s human, and
5.34 s puzzle.  Partial loss maxima were 1.38 s, 2.25 s, 0.45 s, and 0.35 s.

The generic placement minimax exceeded 120 seconds on partial root
`sp_0_p50`.  The measurement harness therefore specializes the same exact
schedule algebra: `A,D,D,A,A,D,D,A` for the current attacker and its loss
dual.  It reduces the last move to singleton-threat cardinality and the loss
endpoint to `forall A-pair, exists D completion-pair`.  Twenty tractable
SecondStone roots were evaluated by both implementations with zero mismatches;
the formerly slow root resolves in about 20 ms for both predicates combined.

## 5. Production recommendation

**HYPOTHESIS.** Port the h6 attacker pair-fork first.  It is millisecond-scale
in Python and fires on 3--7% of fresh self-play/human roots.  The all-phase
measurement shows that a later partial-turn port can add similar attacker bite.

**HYPOTHESIS.** Gate the h8 loss dual.  Human bite is real, but the 33.6-second
Python tail and zero grind hits argue against unconditional evaluation at every
cap-500 leaf until compiled p99/max measurements exist.  The exact integration
and compact `NoWinWithin8` format are specified in the port document.

**CONJECTURE.** The h10 quotient is suitable as an offline label/certificate
tier after a compiled or SAT-style search replaces the Python universal loops.
The theorem has crossed the semantic frontier; this implementation has not
crossed the production-runtime frontier.

## 6. Reproduction, boundary frames, and hashes

Completed commands:

```powershell
python .scratch\horizon_production_h8.py --out .scratch\horizon_production_h8.json
python .scratch\horizon_production_h8.py --validate-existing --out .scratch\horizon_production_h8.json
python .scratch\horizon_h10.py --bounded-audit --out .scratch\horizon_h10.json
```

The complete h8 run took 283 seconds.  The attempted full h10 validation used
one process and timed out at 1,200 seconds.  Two representative sampled h10
negative runs were separately stopped after greater than five and three
minutes; they produced no verdict artifact and are evidence only for the
runtime boundary.

**MEASURED.** SHA-256:

- `.scratch/horizon_h10.py`: `C9BCB97096E5C17BD2A6E294969C44CE12E09D193772BFDC2462E5FEA2BCC4E3`
- `.scratch/horizon_h10.json`: `7A2D580F02323156496DD7C4D323FFB80B48D8795F5E3BE416C5838E4B32FD5F`
- `.scratch/horizon_production_h8.py`: `6F765910DA01BC4CB184EC66E4A1B329102BB9E459605FB71DD7891D5996ED35`
- `.scratch/horizon_production_h8.json`: `CDFB5FA0634D2A879E65F8CE3A6D2035EE7EE4604D30CB5C7F2A92E7AB22F960`
- `docs/PORT_SPEC_HORIZON_H8.md`: `2B7B4DB618404114A63962C3F856231A613F01D0BD1CCC4B7E329598E9AD151B`

No engine, verifier, Lean, or cargo file was edited.  No cargo command or git
commit was run.
