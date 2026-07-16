# GAP-RAW proof round 1 — hostile review

**Target:** `GAP_RAW_PROOF_ROUND1.md` at worktree commit `07560f85`  
**Review disposition:** **REJECTED**

The round does contain useful local facts, most notably K1, L5b, and the narrow
clean-escape calculation.  It does **not** establish its advertised reduction of
GAP-RAW to W3′ + O1′.  Four independent load-bearing failures suffice:

1. Theorem A's right-hand side constrains the debt visible at an epoch but not
   the strategy's actual reply.  L1.2 proves that a servicing reply exists; line
   67 silently changes that to the assertion that the chosen strategy uses it.
2. O1′ is false as quantified.  A finite `Θ₂=0` position with three separated
   count-one promotion gadgets defeats every two-stone defender reply and makes
   `Θ₂>1` on the next attacker turn.
3. K3 does not reduce to a pairwise-overlapping witness family.  The mass
   argument gives matching number at most two, not pairwise intersection, and
   defusing two disjoint representatives does not defuse the witnesses that
   overlap them.  W3′ therefore does not close K3.
4. Theorem D conjoins a statewise W3′ action and a separately existential O1′
   rule without quantifying one strategy that makes both choices.  Its own
   consistency note recognizes, but does not repair, this quantifier failure.

There are also two concrete refutations of L8.3.2's pricing claim, a literal
counterexample to L6, and material defects in the evidence advertised as global
L2/L3 verification.

## 1. Attack #0 — fidelity to the normative gaps

The normative `ES_GLOBAL_BOUNDARY.md` claim is:

> Every **finite, nonterminal blanket-game** position with Defender at
> `FirstStone` and current `Φ<1` has some forever-blocking Defender strategy.

The proof target has the same intended quantifier order `∀P ∃S`, the same
Defender-FirstStone `D,D,A,A` parity, and the same potential:

`λ=√3`, alive/touched means at least one attacker and no defender, and a
count-`s` window contributes `λ^(s-6)=λ^(-e)`.

It is not literally the same statement, however.  D0.2 defines a position only
as `(A,D)` and omits finiteness, nonterminality, placement phase, and blanket
Maker–Breaker semantics.  Calling the game simply "Hexo" also leaves open the
actual-engine rule in which Defender can complete six.  Under that reading,
L1.2's `τ≥3 => Attacker wins` is false when Defender wins on its first placement.
Under D0.2's literal terminal-inclusive reading, a completed attacker window is
still alive, has `e=0`, and has an empty residual set.  Six attacker stones then
give an un-hittable member of `I`, literally contradicting L1.1, L2's seven-stone
floor, and Theorem C's unrestricted opening sentence.

These are repairable domain defects, not a different value of `Φ`.  The required
repair is to import the normative finite, nonempty, nonterminal blanket-state
definition and to say `Attacker-FirstStone`/`Defender-FirstStone` wherever a
two-placement budget is used.

The claimed gap reduction is not faithful:

- `GAP-GLOBAL-RENEWAL` says the existing proofs do not renew the raw hypothesis
  and that `Φ<1` itself cannot be the renewed invariant.  It does not say, as
  proof line 209 does, that "no proof route returns."
- O1′ replaces that gap by renewal from **every** `Θ₂<1` position.  This is a
  strictly stronger statement, and §3 below refutes it.
- `GAP-AMORTIZED-ABANDONMENT` requires a history-sensitive account that remains
  safe when dormant stones are reused.  Deleting count-one mass does not solve
  or subsume that requirement: count-one mass can grow across the grade in one
  placement.  The proposed cross-axis price is also false under the temporal
  interpretation O2′ needs.
- W3′ does not subsume K3, and W3′ + O1′ do not quantify one common strategy.

Thus a perfect proof of the displayed W3′ and O1′ would not presently be the
claimed two-piece proof of the normative boundary.  In fact, displayed O1′ has
no proof because it is false.

## 2. Per-claim verdicts (the target's §5 inventory)

| §5 claim | Verdict | Hostile-review result and exact repair |
|---|---|---|
| L1.1 completion criterion | **REPAIR-NEEDED** | Correct at a finite, nonterminal **Attacker-FirstStone** position.  It is false at `SecondStone`: a count-4 window has `e=2` but only one placement remains.  It is also false literally on an already completed window.  Add both premises. |
| L1.2 epoch service criterion | **REPAIR-NEEDED** | The hitting-set geometry and distance-five legality hold at Defender-FirstStone.  State the quantifier as "Defender has a legal servicing reply iff `τ≤2`."  Spare existence needs a finite nonempty position; board infinitude alone does not prove it.  Restore blanket semantics/nonterminality. |
| Theorem A / A′ exact reformulation | **REFUTED** | Line 67 uses existence of a cover as though the strategy's actual pair were that cover.  The right side permits a strategy to miss a payable threat and terminate in a loss before another defender epoch.  Require the strategy's **actual ordered pair** to hit `I(P)` at every reached epoch, against every attacker continuation.  A′ can then be proved directly; it is not immediate from current A. |
| L2 fork floor | **REPAIR-NEEDED** | The cited search is not global: it enumerates edge-connected polyhexes, and the superadditivity bridge is invalid.  It also tests exact count-4 residual pairs, while L2 quantifies over count-4 and count-5 windows.  Restrict to nonterminal positions and supply a pencil proof or a rigorously complete variable-residual enumeration.  Until then change `VERIFIED` to `UNVERIFIED`. |
| L3 per-`n` ceilings | **REPAIR-NEEDED** | The values are maxima over the connected enumeration, not verified absolute maxima over arbitrary configurations.  The unrestricted cross-check covers only total count-3/4 maxima for `n=4,5,6`, not the four advertised columns for `n=4..12`.  Downgrade the scope or rerun a complete configuration search. |
| L4 one-turn maturation | **REPAIR-NEEDED** | The report supports six named roots, not all `Φ<1` two-stone positions, and does not establish that the called roots are globally "densest."  The inference "at most two pairwise disjoint, hence `τ≤2`" is invalid in general.  For the reported straight four, exhibit the actual two-cell cover.  Item 1's one-stone pencil fact is confirmed. |
| L5a per-root `ΔΦ` ceilings | **CONFIRMED** | The report's per-root table supports `2.309` for the named one-stone roots and `2.591` for the two tested adjacent two-stone roots.  The target correctly avoids the source report's erroneous universal `4/√3` wording. |
| L5b general `ΔΦ` bound | **CONFIRMED** | Attack by count-5 promotion, fresh entry, overlap, and sequential placement all held.  Existing touched mass through the cell gains `(λ-1)S`; at most 18 virgin windows enter with total `2/√3`; iteration gives the displayed bound. |
| L6 service capacity, legality, kill multiplicity | **REFUTED** | "Any empty cell of any alive window" is false because D0.2 also calls a remote virgin window alive.  At `A={(0,0)}, D={(1,0)}`, the window `(100,0)..(105,0)` is alive and every cell is illegal.  Replace with "attacker-touched alive window."  Also covers, not only spares, can have discretionary choices (K1 uses that choice). |
| L7.1 membership; L7.2 decomposition; L7.3 defusal | **REFUTED** | L7.1 and L7.2 held.  L7.3 proves only that `D@c_i` kills cluster `F_i`; it does not prove that one cell annihilates the whole witness.  A witness may have two independent heavy trigger clusters.  Its T2 statement also permits `τ(F₁)=τ(F₂)=2`, so the remainder is not "single-window clusters."  Rewrite L7.3 as a per-cluster fact and explicitly solve the set cover over all heavy trigger cells. |
| L7.4 witness mass floors | **CONFIRMED** | The count cases, distinct-window accounting, count-4/count-5 mixtures, and `1/3` floor held.  K1 uses only this proven floor, not the conjectured `0.415` refinement. |
| R7.4 sharp `0.415` floor / no collinear T1 | **CONFIRMED** | Confirmed only **as honestly labeled CONJECTURED**.  An independent finite attack over all trigger separations 1–5 and all local no-blocker line subsets satisfying I1 found no pure all-count-2 T1.  The same-axis counterexample in §4 has `τ=2`, so it does not refute this narrower T1 conjecture.  The document still lacks the promised exhaustive proof. |
| L7.5 local five-prestone floor | **REPAIR-NEEDED** | The glue from L2 is sound, including the case where only one trigger lies in the footprint.  Its `VERIFIED` strength nevertheless inherits L2's incomplete evidence and terminal-scope defect. |
| Theorem C horizon extension | **REPAIR-NEEDED** | Conditional on a repaired L2, the no-earlier-than induction works for every servicing defender, including one that wastes every spare.  Add `nonterminal`; say completion **can be no earlier than**, not that the bound is attained; define `t*` from 0.  The §5 `a₀≤2` lower-bound wording is the sound strength. |
| L8.1 clean escape / Cor-2 neutralization | **REPAIR-NEEDED** | Exactly zero `Θ₂` on the 36 clean-escape births is confirmed, as is the fact that this particular Cor-2 line supplies no `Θ₂` lower bound.  "Neutralized for graded accounts" is too broad: the grade is not closed under later count-one promotions.  Narrow the label to this direct obstruction only. |
| L8.2 `Θ₂` injection bounds | **REPAIR-NEEDED** | The local formula, exact adjacent-pair `5/9` benchmark, and one-chase `1/9` residue held.  L8.2.2 is a benchmark, not the later claimed universal "remote minting `≤5/9`" bound.  One placement can inject `10/9` (or up to the formula's `2`) from stored count-one windows. |
| L8.3 locality and reactivation pricing | **REFUTED** | Locality around the two trigger cells and per-cell killing hold.  Both the same-axis exclusion and the useful cross-axis two-past-promotion price fail; explicit witnesses are in §4.  Split items 1/3 from item 2 and withdraw the pricing claim. |
| O2′ amortized abandonment | **REPAIR-NEEDED** | Its OPEN label is honest, but its asserted sharpening relies on the refuted L8.3.2 price and ignores cross-grade count-one stock.  Re-state it only after defining a history account that includes latent count-one promotion capacity and a precise time interval for a charge. |
| Theorem B, K1 (`τ(I)=2`) | **CONFIRMED** | Independently rederived for all count-4/count-5 mixtures and overlapping geometries.  `mass(I)≥2/3`; for any cover a ripe pre-witness is label-disjoint from `I` and has mass at least `1/3`, contradicting `Θ₂<1`.  Indeed the argument shows every two-cell cover is unripe. |
| Theorem B, K2/K3 | **REFUTED** | K2's limited mass observation survives: with `mass(I)≥1/3`, all possible surviving witness sets must pairwise share a window.  K3 does not: no three mutually disjoint witnesses means only matching number at most two.  An intersection graph such as `A—B—C`, with `A` and `C` disjoint, is not pairwise intersecting.  Killing chosen representatives does not kill `B`.  The asserted count-3 pool reduction is unsupported. |
| W3′ overlap defusal | **REPAIR-NEEDED** | It is undefined which pre-spare universe of witnesses is being simultaneously defused, and it covers only pairwise-intersecting families, not K3's matching-number-two residue.  The proposed radius-8 locality is also unproved: the two legal trigger cells of one pair can be arbitrarily far apart because each can be legal near old stock.  Formalize and enlarge the obligation before enumeration. |
| O1′ `Θ₂` renewal | **REFUTED** | The three-gadget `Θ₂=0` position in §3 forces `Θ₂≥10/9` after one attacker placement against every defender pair.  Restricting to histories reachable from an original `Φ<1` root under one named strategy is the minimum domain repair; even that repaired obligation remains open. |
| R1b break epochs have `Θ₂≥1` | **REPAIR-NEEDED** | The implication `τ≥3 => Θ₂≥1` is correct.  The cited trace prints numbers of imminent windows but does not compute the minimum hitting set for every stored break epoch.  A fixed policy can lose despite an alternative two-cell cover, so loss alone does not prove `τ≥3`.  Instrument and assert `min_hitting_set≥3` on every claimed epoch. |
| Corollary B′ / Theorem D | **REFUTED** | W3′ does not close K3, and the W3′ action is not jointly quantified with the O1′ rule.  If O1′ is read as a complete servicing strategy, it alone forces unripeness and W3′ is redundant; if read as a spare prescription, D's conjunction is invalid.  Replace both assumptions by one joint strategy obligation. |
| No forced pileup/six, plies ≤6 | **REPAIR-NEEDED** | Confirmed only for the six named sibling-report roots (`es_core`, `blocker_1_-1`, `blocker_2_0`, `blocker_3_0`, `dense_01_10`, `dense_01_20`).  It is not universal over `Φ<1` roots.  State that scope and retain the report's horizon/cap qualifications. |

## 3. Concrete refutation of O1′ and closure of the graded account

O1′ says one rule maintains `Θ₂<1` from every `Θ₂<1` start.  Use centers

```text
c0 = (6,-3),   c1 = (18,-3),   c2 = (30,-3).
```

At every `ci=(q,r)`, put attacker supports at `(q+1,r)` and `(q,r-1)`.  This is
not merely an arbitrary blanket position: it occurs at Defender-FirstStone after
the following legal, nonterminal history (the first D placement is the opening):

```text
D (0,0)
A (7,-3), (6,-4)
D (12,0), (13,1)
A (19,-3), (18,-4)
D (24,0), (25,1)
A (31,-3), (30,-4)
```

No two attacker stones share a length-6 axis window: the two supports in one
gadget differ by `(1,1)`, which is not one of the three axes, and different
gadgets are more than five apart.  The old defender stones kill two remote
count-one labels but none of the focal labels below.  The exact alive profile is
106 count-one windows and no count-two-or-higher window, hence

```text
Θ₂ = 0,    I = empty.
```

At `ci`, the first support belongs to five Q-axis windows through `ci`; the
second belongs to five R-axis windows through `ci`.  These are ten distinct
count-one windows.  The unions of the ten focal windows for the three centers
are pairwise disjoint.  Two defender spares can therefore meet at most two of
the three unions, even if they occupy two centers outright.

Attacker chooses the untouched center and places there legally (it is adjacent
to both supports).  All ten focal windows become count two, so after the first
attacker placement

```text
Θ₂ >= 10 * (1/9) = 10/9 > 1.
```

The second attacker placement cannot decrease `Θ₂`, and no completion occurs,
so the violating next Defender epoch exists.  This refutes O1′ against every
defender rule, not merely a greedy rule.

It also pinpoints the closure error requested in the review order: mass below
the grade is not inert.  The account operations in §3.8 do not include the
latent capacity of count-one windows to enter at weight `1/9`.  L8.2.1 permits
this (`n₁(c)/9`); line 231 then incorrectly replaces it with the isolated-pair
benchmark `≤5/9`.

## 4. Concrete refutations of L8.3.2

### 4.1 Same-axis heavy formation costs one past placement

On a Q-axis take old local stones `{-2,-1}`, then place `p=1`.  Let the
prospective trigger be `c=0`.  The three windows starting at `-4,-3,-2` were
count two before `p`, are count three after it, and after a future `A@c` have
residual pairs

```text
{-4,-3},  {-3,2},  {2,3}.
```

Their hitting number is two.  Thus one past promotion creates a same-axis heavy
cell without a pre-existing count-4 window.  To embed it in an actual ripe T2
witness, add remote count-three stock at

```text
(100,20), (102,20), (105,20)
```

and use the second future trigger `(101,20)`.  That remote window contributes a
residual pair disjoint from the local family.  The full future family has four
windows and hitting number three, while Q satisfies I1.  This directly refutes
the sentence that the worked R7.4 cases exclude one-placement same-axis heavy
formation.  R7.4 considered the narrower pure-collinear **T1 (`τ≥3`)** case;
heaviness requires only `τ≥2`.

### 4.2 The cross-axis proof omits the mixed count-2/count-3 branch

Let

```text
Q.A = {(-4,0), (-3,0), (0,1), (0,2), (0,3)},
c = (0,0), d = (1,0).
```

Q satisfies I1.  The horizontal window `(-4,0)..(1,0)` is count two and
contains both future triggers.  Three vertical windows through `c` are count
three.  After the pair `(c,d)`, the four resulting residual pairs have hitting
number three.  Remove the single past placement `p=(0,3)`: every one of these
four windows is then count at most two.  Thus one past placement, not two,
creates this cross-axis heavy-in-formation witness.

L8.3.2's proof considers only two cross-axis windows that both had to be
promoted to count three.  L7.1 explicitly permits the omitted case: a window
containing both future triggers may remain count two and jump directly to count
four.  If the phrase "after last count ≤2" is instead read to count the future
trigger pair itself, the lower bound becomes vacuous—every witness has those two
placements—and cannot pay for the defender spare that must pre-empt the witness.
Either reading fails the use asserted in O2′.

## 5. Theorem A, legality, and game-tree quantifiers

### 5.1 The action/epoch swap

Here is a local board that isolates the bad inference at line 67.  Put

```text
A = {(i,0): i=0..4}
D = {(-1,0),(6,0)}
    union {(i,-2),(i,1),(i-2,2),(i+1,-1): i=0..4}.
```

There is exactly one attacker-touched alive window, `W={(0,0)..(5,0)}`.  It is
count five, so `Φ=1/√3<1`, `I={W}`, and `τ(I)=1`; Defender has no completed six.
The legal defender pair `(0,3),(0,4)` does not service W, after which legal
`A@(5,0)` wins.  This does not refute L1.2's existence of a good reply.  It
refutes the proof step that uses that existence to certify the actual strategy.

The repair must put the action in Theorem A's right-hand side:

> There is a Defender strategy S such that, against every attacker
> continuation, at every reached Defender-FirstStone epoch its actual ordered
> pair is legal and hits every member of `I(P)` (equivalently, it restores I1).

With that repair, a finite bad prefix is enough for each direction.  No hidden
determinacy or König assumption is needed.  Every finite position has a finite
legal move set (a finite union of radius-8 balls), although the branching bound
grows with time.  The proof fixes one strategy and never passes from
horizon-dependent strategies to an infinite one.  König is needed only for a
finite-horizon compactness argument such as boundary Theorem 4, where it is
already stated.

### 5.2 Legality attacks

- Every empty of an imminent window is legal: it is within axis distance at
  most five of one of that window's at least four attacker stones.  This attack
  held.
- Every empty of an **attacker-touched** alive window is likewise legal.  The
  unqualified L6 version fails on remote virgin windows.
- Spare placements exist only after finiteness/nonemptiness is restored.  A
  clean proof chooses an extremal occupied cell in a lattice direction; an
  outward neighbor is empty and legal, and the argument repeats for the second
  placement.  "The board is infinite" by itself does not exclude an infinite
  occupied position with no legal empty.
- Color-blind radius-8 legality matches the engine rule.  The defect is the
  missing state premise, not the metric.

## 6. K1, K2/K3, W3′, and Theorem D

### 6.1 K1 survives hostile arithmetic

The `τ(I)=2` argument handles every mixture of count-4 and count-5 windows and
does not require them to be disjoint.  `τ=2` implies at least two distinct
windows, each weighing at least `1/3`, so `mass(I)≥2/3`.  After a two-cell cover,
any ripe witness consists of P-alive count-2/count-3 window labels, hence is
disjoint as a label set from I and has pre-mass at least `1/3`.  The total would
be at least one.  K1 is sound—and proves every cover good, not merely some cover.

### 6.2 K3 has another unhandled case

For K2, the existing `mass(I)≥1/3` means any two possible witness sets that were
window-disjoint would force total mass at least one.  Its residual family is
therefore pairwise window-intersecting.

For K3, the absence of three mutually window-disjoint witnesses says only that
the witness hypergraph has matching number at most two.  It does **not** say the
family is pairwise intersecting.  The simplest missing intersection graph is

```text
A -- B -- C,       A disjoint from C.
```

Choosing heavy cells for A and C proves only that those two witnesses are
defused.  No lemma says B contains either chosen heavy cell.  More generally,
overlap of window labels is not overlap of trigger/heavy cells.  The sentence
about a pool of at most two count-three windows is an additional unsupported
claim, and killing one pool window need not lower a `τ≥3` cluster below
heaviness.

This is exactly the requested "other unhandled case."  The proof has not
cornered K3 into W3′.  A geometry theorem excluding the path case would be new
load-bearing work, not a detail already present.

L7.3 introduces a second K3 problem: a single witness pair may have two remote
heavy halves.  For example, around a cell `c` use the translated nine-stone set

```text
{(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(-1,1),(-2,2),(-3,3)} + c.
```

Before `A@c`, every alive window has count at most three; after it, the local
imminent family has hitting number six.  Two far-separated copies, triggered by
the two cells of one legal attacker pair, give a T1 witness for which `D@c1`
kills only the first half and leaves the second half unblockable.  The true
object to cover is the global set of heavy trigger cells, not one chosen heavy
cell per witness.

### 6.3 Theorem D does not compose its assumptions

W3′ says a suitable spare action exists at a state.  O1′ says some adaptive rule
maintains an inequality.  `∃a W(a)` and `∃S O(S)` do not imply that S chooses a
W-good action.  This conflict also exists among two-cell covers on `τ=2` turns:
K1 certifies unripeness, not compatibility with a renewal choice.

There are only two coherent readings:

1. If O1′ already means a complete legal **servicing** strategy that maintains
   `Θ₂<1` against every attacker continuation, then it automatically hands over
   only unripe positions: a ripe pair would create a next epoch with `τ≥3` and
   hence `Θ₂≥1`.  W3′ is then a construction lemma, not a second theorem
   assumption.
2. If O1′ prescribes only spare/account choices, Theorem D needs a separate
   joint-choice hypothesis, which it lacks.

Replace the two assumptions by one strategy statement J, quantified only on
histories reachable from the original `Φ<1` roots, whose actual moves jointly
cover I, hand an unripe position, and renew the chosen account.

## 7. Evidence and label audit

### 7.1 L2/L3 are not globally verified by the cited harness

The geometry test enumerates only edge-connected polyhexes.  The report's
bridge—superadditivity of the connected maximum implies a connected global
optimum—is invalid because edge-disconnected components need not contribute
independently.  On one Q-line, for example,

```text
{(0,0),(2,0),(3,0),(4,0)}
```

is edge-disconnected but the components jointly occupy four cells of a single
length-6 window.  The objective is not the sum of component objectives.

The `[0,6]^2` unrestricted brute force checks only total count-3/count-4 maxima
for `n=4,5,6`.  It does not check the fork/hitting-set predicate, count-5
residual singletons, or any advertised ceiling at `n=7..12`.  Moreover,
`count4_empty_pairs` excludes count-5 windows while L2 uses all `e≤2` windows.

Finally, the ignored geometry and maturation tests print `gen_ok`,
superadditivity, brute-match flags, and numerical tables but assert none of
them.  A Cargo `PASS` proves only that the print-only test returned normally;
it is not by itself an assertion of the advertised vector.

This review did not find a ≤6-stone nonterminal counterexample to L2; the point
is that its stated global verification has not been performed.  The likely
repair is a short pencil reduction (different-axis count-4 windows already need
seven stones) plus a complete one-dimensional residual-cover lemma, or an
unrestricted co-window-connected enumeration with count-5 residuals and hard
assertions.

### 7.2 Matching number is not hitting number

The reports repeatedly use "at most two pairwise-disjoint residual pairs" as if
it implied a two-cell cover.  It does not.  The six edges of `K4`, for example,
have maximum matching two and vertex-cover number three.  The geometry may
exclude that graph in a particular row, but the exclusion must be proved.

For the specific straight-four claimed in L4, the repair is easy: with attacker
stones at `0,1,2,3`, the three residual pairs are

```text
{-2,-1}, {-1,4}, {4,5},
```

and `{-1,4}` is an explicit cover.  The current implication is nevertheless an
invalid proof step.

### 7.3 Citation scopes

- L4/L5a are per named root, not universal.  "Densest two-stone roots" is not an
  exhaustively established classification.
- The sibling pileup result is exhaustive through six plies for six named
  roots, not all `Φ<1` positions.
- Boundary Theorem 2 proves five-placement safety.  Its sharpness example is
  for the specified fixed-family strategy **with its specified filler**, not
  the entire fixed-cohort strategy class as line 177 says.
- The R1b trace does not calculate `τ` on every break epoch.  Count of windows
  and loss of one fixed policy do not substitute for the minimum hitting set.
- The target's provenance is stale: current HEAD is `07560f85`, not
  `9b32db63`.  The local and sibling harness files are byte-identical (SHA-256
  `ADC6C9AA745D06CEC709B93A582A712B11375DDCE06E3BEF96E87D32E4727886`), so this
  is a labeling repair rather than a source-divergence defect.

## 8. Machine attacks run in this review

I ran a deterministic independent window enumerator over the three axial
directions.  It made no random choices and wrote no source files.  Its asserted
results were:

```text
O1 gadget: reachable initial profile {count1:106, count2..6:0};
           10 promotions at each center; focal unions pairwise disjoint.
Cross-axis L8.3 witness: pre-max count 3; base before one promotion max count 2;
                           future family size 4, tau 3.
Same-axis L8.3 witness: pre-max count 3; future family size 4, tau 3.
R7.4 attack: all trigger separations 1..5 and all local I1 subsets;
             zero pure all-count-2 collinear T1 counterexamples.
Theorem-A local board: exactly one alive touched window, count 5; no Defender six;
                       both wasted moves and the winning completion were legal.
```

The collinear run is an attack result, not a replacement for the missing
documented proof: it covered the advertised local no-blocker line model and
left R7.4 plausible.  No Cargo run was needed; free physical RAM was checked at
approximately 12.8 GiB and there was no concurrent Cargo process.

## 9. Overall disposition

**REJECTED.**  This is not installable with a list of editorial corrections.
The exact reformulation is missing its action quantifier, O1′ is refuted, the
L8.3 price is refuted, K3 has unhandled non-pairwise families, and Theorem D
does not bind one joint strategy.  Those defects invalidate the central
"reduced to exactly W3′ + O1′" result.  L2/L3 must also lose their global
`VERIFIED` labels until their configuration universe and residual family are
repaired.

## 10. Priority-ordered repair list for round 2

1. **Restore the normative domain verbatim.**  Define finite, nonempty,
   nonterminal blanket Maker–Breaker positions with explicit `FirstStone` phase;
   exclude completed windows before defining `I` and `τ`.
2. **Replace Theorem A's right-hand side.**  Quantify one strategy whose actual
   legal two-cell reply services `I(P)` at every reached defender epoch against
   every attacker continuation.  Prove A′ directly from that statement.
3. **Withdraw O1′ as stated and add the three-gadget regression.**  Replace
   "every `Θ₂<1` start" with histories reachable from an original `Φ<1` root
   under the same named strategy; do not claim renewal until count-one latent
   promotion capacity is included.
4. **Replace W3′ + O1′ by one joint obligation J.**  J must select the actual
   cover/spare pair, hand an unripe position, and renew the account on every
   J-reachable history.  Do not conjoin separate existential witnesses.
5. **Delete L8.3.2's pricing claim.**  Add both explicit counterexamples above;
   case-split pre-count-3/pre-count-3 from the mixed pre-count-2/pre-count-3
   branch; define exactly which past interval any reactivation charge covers.
6. **Rebuild K3 before naming its residue.**  Either prove a two-cell defusal
   theorem for every witness collection of matching number at most two, or prove
   a geometry lemma reducing path/cycle intersection graphs to the pairwise
   case.  Remove the unsupported count-three-pool sentence.
7. **Rewrite L7.3 as per-cluster only.**  Explicitly allow two heavy halves of
   one witness and formulate suppression as a hitting problem on all heavy
   trigger cells.
8. **Reverify L2/L3 over a complete universe.**  Include edge-disconnected but
   co-window-interacting configurations, count-5 singleton residuals, all four
   L3 columns, and hard assertions for every advertised number.  Until this is
   done, label L2/L3/L7.5/Theorem C as dependent on an unverified lemma.
9. **Repair the straight-four cover step.**  Replace matching-number language by
   the explicit two-cell cover and restrict L4/L5a to the exact named roots.
10. **Instrument every R1b break epoch.**  Print and assert the variable-size
    minimum hitting set, then infer `Θ₂≥1` only on epochs where `τ≥3` is actually
    established.
11. **Narrow all evidence prose.**  Say "six named roots," narrow Theorem 2
    sharpness to its specified strategy/filler, narrow Cor-2 neutralization to
    its direct clean-escape source, and update the HEAD/provenance labels.
12. **Correct Theorem C's statement.**  Add nonterminality and use "can be no
    earlier than" throughout; do not claim attainability from the L2 lower bound.

## 11. Independent judgment on W3′ and O1′

**W3′:** the narrowly intended, precisely formalized pairwise-intersecting case
looks **provable**, probably by finite geometry plus the `<1` mass budget, but
the displayed statement is ill-defined and is insufficient for K3.  The
sharpest attack is a witness intersection graph of matching number two that is
not a clique (`A—B—C` or a longer path/cycle), combined with the fact that shared
window labels need not share a defusal cell.  Any enumeration must also handle
two arbitrarily separated trigger neighborhoods; the proposed one-center
radius-8 search is not yet justified.

**O1′:** **refutable as written**, by the three separated count-one gadgets in
§3.  A repaired, strategy-reachable joint invariant is genuinely open and is
likely independent of the present finite geometry facts: neither L8.1 nor K1
controls the latent `n₁(c)/9` capacity that crosses into `Θ₂`, and no existing
lemma couples renewal choices to W3′ defusal choices.
