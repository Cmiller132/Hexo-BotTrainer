# Dispatch-domination hostile review — round 1

## Verdict

- Overall verdict: **ACCEPT-WITH-EDITS**
- L-DISPATCH-B1, `n = 0`: **CONFIRMED**
- L-DISPATCH-B1, `n = 1`: **CONFIRMED**
- L-DISPATCH-B1, `n >= 2`: **CONFIRMED**
- L-DISPATCH-B1, every-legal-defender-reply strengthening: **CONFIRMED**
- L-DRQ: **CONFIRMED**
- Solver-consumption safety: **CONFIRMED for the stated `b = 1`, formal-rule
  scope; not a certificate for `b = 2`, byte-state merging, or unchecked
  coordinate overflow**

No explicit counterexample survived the hypotheses of either PROVEN theorem.
The required edits are to the `b = 2` experimental design, provenance, and
consumption wording, not to either pencil proof.

## Findings by severity

### Critical

None. In particular, I found no false solver-facing PROVEN claim and no
transitive route from L-DISPATCH-B1 to coverer interchange.

### Major

1. **The mandatory `b = 2` bulk/audit depths are tautological for covered
   candidates.** Every covered successor is exactly `Unknown` at `d = 0,1,2`.
   The nontrivial conjectures can first separate at `d = 3`, and quiet/frontier
   effects may need later depths. The current plan makes those depths optional
   and potentially incomplete while describing `d = 1,2` as substantive
   cross-product adjudication.
2. **Several pre-registered controls can be vacuous or misclassified.** The
   K1 panel selects only one dead spare, so it need not contain a DRQ pair. The
   K2 panel does not guarantee a P2-protected substitution. The lifted b=1
   coverer witnesses are not automatically K1 hit-any cases at the preceding
   `FirstStone` parent: both recorded cells must still belong to that parent's
   `H(P)`, and the actual corpus first stone may have killed additional parent
   threats. The design needs explicit eligibility assertions and `NOT TESTED`
   outcomes rather than treating these as assured controls.

### Minor

1. The proof is committed at reviewed HEAD `7e240388`, but its banner names
   stale commit `6b853c0e`.
2. Section 7.2's phrase "answer about the b=2 dispatch arm" conflates the
   primary K1 spare case (`mhs = 1 < b = 2`) with shipped `implicit_dispatch`,
   whose b=2 arm is the distinct `mhs = b = 2` path.
3. Panel 3 fixes the spare `s` and varies the hit; it attacks hit identity and
   hit/spare interaction, not fixed-hit spare identity as claimed.
4. The pencil rules use `Z^2`, whereas production coordinates are unchecked
   `i16`. The proof acknowledges the inherited no-overflow qualification,
   but a production consumer still needs that domain condition enforced or
   separately discharged. This is a global carrier limitation, not a local
   counterexample to either theorem.

## Mandatory attack-surface audit

### A. L-DISPATCH-B1 short horizons

The short-horizon strengthening is sound because the definitions are global,
not witness-local. `T_A(P)` is the complete family of all current attacker-
alive count-4/5 windows, and a full-coverer hits **every** member. Therefore
an alleged "other count-5 window" left unblocked proves only that the reply was
not a full-coverer and violates DB1-A.

The exact stopped values are:

| Horizon after reply | Full-coverer `a` | Non-coverer `c` | Reason |
|---|---:|---:|---|
| `n = 0` | `0` | `0` | DB1-NOWIN at b=1 excludes every defender count-5, so neither reply can be `D@0`. |
| `n = 1` | `0` | `0` or `-1` | Any attacker win in one placement would come from a pre-existing alive count-5; `a` kills every such window. |
| `n >= 2` | at least `-1` | exactly `-1` | `c` leaves a count-4/5 window with one or two empties for the attacker's uninterrupted two-stone turn. |

At `n = 0`, `?` has formal utility zero. At `n = 1`, a defender stone cannot
create a new attacker count-5, and a count-4 needs two attacker placements.
This discharges the mandatory other-window attack.

### B. Every-legal-reply strengthening

Confirmed, but the quantifier must be read exactly. The proof does **not** say
that the attacker's two-placement win survives after every compared reply
`r`. It proves the win only after the fixed DB1-C non-coverer `c`, obtaining
`V_D^n(P+c) = -1` for `n >= 2`. Since `-1` is the minimum possible value,
every legal reply `r` satisfies

```text
V_D^n(P+r) >= -1 = V_D^n(P+c).
```

A reply that creates a defender count-5 fork, a forced defender win on the
next turn, or even an immediate defender win only raises the left side and
makes the domination inequality stronger.

I attacked this with the explicit reachable replay recorded below. Its
full-coverer creates four distinct defender count-5 threats and wins for the
defender at further placement 3. The non-coverer nevertheless loses to the
attacker at further placement 2. This is the sharp requested counterfork
case, and it confirms rather than reverses the inequality.

### C. Exact turn structure

The engine transition is:

```text
P: D SecondStone
  time 0: D plays compared reply
  time 1: A FirstStone
  time 2: A SecondStone
  time 3: D FirstStone
```

`state.rs:309-333` checks a win after each placement and advances from
`SecondStone` to the other player's `FirstStone` only on a nonterminal reply.
Thus a missed count-5 wins at time 1 and a missed count-4 wins no later than
time 2. A defender threat created at time 0 cannot fire at time 3 because the
attacker's terminal event at time 1 or 2 stops the trace. There is no
turn-boundary off-by-one.

### D. L-DRQ frontier inertness

The proposed frontier-asymmetry counterexample is impossible under
DRQ-CELLS. For an empty dead cell `b`, each directed endpoint window

```text
{b, b+u, ..., b+5u}
```

is dead and hence contains an old stone `s = b + k u`, `1 <= k <= 5`, on
each of the six axial rays. For any `z in B_8(b)`, choose its 60-degree sector
and write `z-b = alpha*u + beta*v` with `alpha,beta >= 0` and
`alpha+beta <= 8`. Using the ray witness `s` gives

```text
d(z,s) = max(|alpha-k|, beta, |alpha+beta-k|) <= 8.
```

So every point of `B_8(b)` was already supported. A small independent
enumeration of all `5^6` choices of one witness distance per ray found zero
failures (the worst nearest-witness distance was 7). Deadness therefore
implies both legality and frontier inertness under the actual radius-8 rule.

Applying P1 with searched `x`, discarded `y`, then swapping them, is valid.
At a `FirstStone` parent P1 transposes the stored `SecondStone{first}` payload;
at a `SecondStone` parent both replies advance to the opponent's
`FirstStone`. Both special cells are already occupied where the stored-first
non-reuse rule matters. No phase or occupancy asymmetry remains.

### E. Value convention and stopped-horizon semantics

Lemma 3 is used within its stated preconditions: each comparison uses two
legal replies from the same finite, nonterminal parent; the stopped trees are
finite; and the strategic defender role is fixed (or, for DRQ, obtained by a
valid colour-symmetric rename of the current mover).

The formal `?` in DOMINATION.md is an exact no-terminal-through-horizon
outcome with utility zero. It is not a resource cutoff. L-DISPATCH-B1 never
uses a capped solver's `Unknown` as though it were a loss or draw. Likewise,
Section 7's rank convention is correctly oriented because
`solve_for_player(C_M, attacker, d, ...)` returns status relative to the
attacker:

```text
attacker Loss > attacker Unknown > attacker Win
```

for the defender. The proposed experiment correctly separates a completed
exact `Unknown` from `INCOMPLETE`; hunt-side `att_unknown` evidence is not used
to prove either theorem.

### F. Scope honesty (`b = 1` versus `b = 2`)

No actual B1-to-b2 theorem leak was found. The shipped predicate is shared:
`tss_solver.rs:2510-2513,3586-3589` activates `implicit_dispatch` whenever
`min_hitting_set == b`, so it has both b=1 and b=2 paths. But the proof's
engine corollary (`:329-337`) and attack surface (`:887-890`) explicitly
restrict L-DISPATCH-B1 consumption to b=1 and disclaim the b=2 arm.

The code also keeps the mechanisms distinct. At b=1 the kernel is the
intersection of all threat-empty sets. At b=2 the kernel contains cells
extendable to a two-cell transversal (`tss_solver.rs:4954-4974`), and the
pair plan/verifier independently enforce the b=2 conditions. Nothing in the
reviewed proof authorizes collapsing different coverers or hitting sets.

Section 7 should nevertheless rename its K1 `mhs=1<b=2` target as a proposed
**spare-stone pruning** rule, not the shipped b=2 `implicit_dispatch` arm.

## Additional attacks

### Canonical coverer-interchange witness

The canonical `d7e1b56c925b7f32:20` witness remains fully compatible with
both the theorem and its every-reply strengthening. If `a_1` and `a_2` are
the two coverers and `c` is a non-coverer, the theorem supplies

```text
c <= a_1    and    c <= a_2.
```

Two elements sharing a lower bound need not be comparable. DB1-C prevents
reusing either coverer as `c`, so neither symmetry nor transitivity yields
`a_1 <= a_2`, `a_2 <= a_1`, or equality. The opposite proven coverer
outcomes are therefore a boundary witness, not a refutation.

### Machine-evidence scope

The proof reports the hunt honestly. The 20,495 DISPATCH comparisons used one
deterministic full-coverer and only G3 counter-threat non-coverers; they did
not enumerate every legal non-coverer against every coverer. The DRQ hunt had
5,133 pair firings but only 288 adjudicated pairs, admitted
`Unknown/Unknown` as agreement, and had no wide confirmation stage. Neither
sample is promoted to exhaustive verification; the all-reply/all-phase
quantifiers come only from the pencil arguments. Conversely, the four
doubly-proven coverer-interchange failures remain active boundary evidence
and are not diluted by the zero-refutation dismissal sample.

### Rule-outcome pruning is not state merging

L-DRQ equates formal stopped outcomes only. The successors retain different
placement histories, `last_turn` data, model features, and potentially cache
or certificate serialization. A consumer may omit a rule move while still
applying the retained representative through the real phase machine; it may
not merge arbitrary byte states or TT keys.

### The `b = 2` bulk depths are provably non-discriminating

Section 7's candidate comparisons all use a completed pair that covers the
complete initial family `T` of attacker-alive count-4/5 windows.  Such a pair
kills every member of `T`.  Defender placements cannot create a new
attacker-alive count-4/5 window, so at the resulting attacker-`FirstStone`
child every attacker-alive window has count at most 3.  During oracle depths
`d = 1` and `d = 2`, only the attacker moves, once or twice.  It therefore
cannot reach six, and the defender cannot win because the defender has no
placement.  Consequently

```text
O_0(M) = O_1(M) = O_2(M) = Unknown
```

for **every covered candidate `M`** under Section 7's predicate.  (Uncovered
B2-COVER controls are the intended exception.)  Thus the claims in Section
7.5 that `d = 1` catches immediate attacker-completion distinctions and that
the `d = 1,2` universal audit substantively adjudicates the covered-pair
subclaims are misleading.  The first possible covered-child distinction is
at `d = 3`, when the defender gets a placement; quiet/frontier distinctions
may require still more depth.

This does not attack either round-1 PROVEN theorem.  It does mean the `b = 2`
experiment has genuine falsification power for the nontrivial subclaims only
to the extent that its optional deeper solves complete.  The mandatory bulk
and cross-product depths by themselves test a tautological equality baseline.

## Repair list

1. Replace the stale `6b853c0e` banner with the actual reviewed proof commit
   `7e240388` (or state clearly what the older hash denotes).
2. Add the analytic baseline `O_0=O_1=O_2=Unknown` for every covered b=2
   candidate. Recast `d=1,2` as uncovered controls/smoke tests, not
   discriminatory evidence.
3. Make at least `d=3` mandatory for every nontrivial covered-candidate
   comparison, and state which later depth is mandatory for quiet/frontier
   classes. If the complete audit cannot finish that depth, report the
   associated directional subclaim as `NOT ADJUDICATED`.
4. Give DRQ-LIFT and K2-P2-LIFT explicit eligible-pair quotas, or require a
   manifest count and report `NOT TESTED` when it is zero.
5. For each lifted coverer witness, assert at the preceding parent that
   `!own_win_now`, K1, and both recorded cells are in `H(P)`; retain the P3
   reverse aliases. Otherwise label it a generic stress fixture, not a
   HIT-ANY control. Do not promise a mismatch by `d<=6` when the historical
   solve used a horizon up to 40.
6. Correct the K1/shipped-b2 terminology and Panel 3's spare-identity wording.
7. Preserve the explicit b=1 consumption fence and make the inherited
   coordinate-carrier precondition enforceable at the production boundary.

## Independent assessment of the `b = 2` experiment design

The pre-registered logical falsifiers are correctly oriented, but the
execution design does not guarantee a meaningful test of every subclaim.

| Subclaim | Is its stated decisive pattern a valid refuter? | Design judgment |
|---|---|---|
| B2-COVER | Yes | Sound analytic/oracle control; uncovered cases can differ at `d=1,2`. |
| B2-P3-ORDER | Yes | A matched-depth mismatch refutes P3 handling if both orders are independently replayed before quotienting. |
| K1-FIRST-DISMISS-INDEXED | Yes | `F_d(c) > max_h F_d(h)` is the correct strict refuter, but only a complete second-move audit can establish it. The mandatory `d=1,2` audit is tautological. |
| K1-CONTAIN-INDEXED | Yes | Requires the complete H-containing universe; sampled panel comparisons cannot refute existence of an untested `M`. |
| K1-CONTAIN-UNIFORM | Yes | An empty intersection over even a finite tested depth set refutes one-`M`-for-all-depths, provided every `M` was enumerated at each included depth. |
| K1-SPARE-ANY-EQ / HIT-ANY-EQ | Yes | One exact mismatch suffices, but the scheduled shallow matrix cannot contain one for covered children; Panel 3 varies hits, not spares. |
| K1-DRQ-LIFT | Yes | A mismatch attacks DRQ/harness, but the selected manifest may contain no pair of dead spares. |
| K2-HSET-ANY-EQ | Yes | One unequal exact pair refutes it; meaningful depth still begins at `d=3`. |
| K2-P2-LIFT | Yes | A mismatch is decisive, but no P2-protected case is guaranteed to exist in the selected panel. |

Accordingly, the design **can** falsify each conjecture if it contains an
eligible case and completes a discriminatory depth. It cannot validate a
general universal (which the proof correctly acknowledges), and its currently
mandatory `d=1,2` core cannot falsify any nontrivial covered-pair conjecture.
Without the repairs above, DRQ/P2 controls may be vacuous and the directional
claims may remain unadjudicated behind optional incomplete deeper solves.

## Evidence and reproduction notes

### Exact placement rule established from production code

The exact non-opening rule at reviewed HEAD `7e240388` is:

1. the state must be nonterminal;
2. the selected coordinate must be empty;
3. on `SecondStone`, it must differ from the stored first coordinate; and
4. it must be in the incremental legal store, which is the union of closed
   radius-8 hex balls inserted after **every** existing stone, independent of
   owner.

The controlling sources are
`packages/hexo_engine/rust/src/legal.rs:17-18,114-145`,
`packages/hexo_engine/rust/src/board.rs:83-105,167-170`,
`packages/hexo_engine/rust/src/rules.rs:10-45`, and
`packages/hexo_engine/rust/src/state.rs:289-357`. Thus the engine radius is
exactly 8. The DTW hunt
report's phrase "within distance 5 of stones" is a sufficient geometric
certificate for cells in one six-cell window, not a statement of the exact
rule; the sibling final proof itself makes that distinction explicitly.

For both dispatch proofs, every required threat-window empty is within
distance at most 5 of an old attacker stone and therefore was inserted into
the radius-8 store before the compared reply.  This legality step does not
depend on which player owns the supporting stone, though here attacker stones
are available directly.

### Explicit counterfork stress replay for L-DISPATCH-B1

With `A=Player0`, `D=Player1`, semicolons separating turns, the following
legal replay ends after D's first stone of the current turn:

```text
(0,0);
(2,1),(2,2);
(1,0),(4,0);
(2,3),(2,4);
(5,0),(-6,0);
(3,-1),(4,-2);
(-6,2),(-4,6);
(5,-3),(6,-4);
(0,7),(7,0);
(0,-7)
```

The resulting D-`SecondStone` node has one A count-4 threat, the Q-axis
window starting `(0,0)`, with empties `(2,0),(3,0)`. D has no count-5, so
DB1-NOWIN holds, and the full-coverers are those two empties.

- Non-coverer `c=(0,1)` misses the window. A fills `(2,0),(3,0)` and wins at
  further placement 2.
- Full-coverer `r=(2,0)` kills it and simultaneously creates four D count-5
  windows with distinct completions `(2,-1)`, `(2,5)`, `(1,1)`, and `(7,-5)`.
  A can block only two; D wins at further placement 3.

The stopped values are therefore

```text
c: 0, 0, -1, -1, ...
r: 0, 0,  0, +1, ...
   n=0  1   2   3
```

This position was chosen specifically to make the compared reply's defender
counterfork as strong as possible. It does not break the theorem.

### Review execution

No cargo grind was needed, no test module was added, and no production or
proof source was edited. The review used source inspection, explicit replay
audits, the existing machine-evidence records, and a small standalone
`5^6` geometry enumeration for the DRQ frontier attack.

### Review provenance

The worktree and the commit containing the proof are `7e240388`.  The proof's
opening banner instead says "Final dispatch, commit `6b853c0e`".  This is a
stale provenance label to repair even if the mathematical claims survive.
