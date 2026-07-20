# Hostile review of the tightness, pairing, and ES reports: round 1

Date: 2026-07-14.

The three supplied checkers were run with ordinary, assertion-enabled Python.
They all exited successfully. The proofs below were re-derived independently;
a passing checker was not treated as proof of a prose claim that the checker
does not encode.

Verdicts in this report have the following meanings. `CONFIRMED` means that
the stated claim and its advertised scope survive. `REPAIR` means that the
underlying result survives only after the exact qualification or checker
change given below. `REFUTED` means that the claim in its present form is
false; it does not mean that every nearby weaker claim is false.

## 1. Tightness report

### 1.1 Verdict table

| Item | Verdict | Finding |
|---|---|---|
| R1a, exact role band `8(r-1)` | **CONFIRMED** | The rank-two trace attains the L9-prime distance-chain inequality. It is a local exact-rank trace, not a complete D9 false-WIN certificate. |
| R1b, uniform band `8(B-1)` | **CONFIRMED** | `OPEN` is the correct ruling. Exact rank need not equal the scalar D14 budget in a complete certificate. |
| R2, virgin radius `8(E^D-6)` | **CONFIRMED** | The fixed-window arithmetic is attained for every `E^D>=6`; sharpness of the full union `Z_virgin` is not proved. `OPEN` in general is correct. |
| R3, touched equality `cnt_D+E^D=6` | **REPAIR** | The coordinates attain the raw count and parity bound. They do not instantiate a complete recurrence-derived D16 certificate, so the sentence that they falsify L12 is too strong. |
| R4a/R4b, LOSS caps | **CONFIRMED** | The `3/5` theorem is correct. The prior cap six improves to five, and the triangle and `C5` Hexo constructions are sharp. |
| R5a, kernel scope `mhs<=b` | **CONFIRMED** | For `mhs>b`, `K_b` is empty. This is a grammar/kernel pin, not an absolute game counterexample. |
| R5b, kernel `not own_win_now` | **REPAIR** | The game counterexample is correct, but its root also fails D9's retained internal-AND diagnostic. It pins removal of all enforcement of the predicate, not deletion of the T6 premise alone. |
| R5c, residual threshold `b-1` | **CONFIRMED** | Two disjoint singleton threats at `b=2` attain residual transversal number one after the first minimum-transversal reply. |
| R6, LOSS condition `tau(T)>b` | **REPAIR** | Equality refutes weakening the combined survivor contract. It is not a certificate satisfying D9's separately written universal-survivor clause. |
| R7, D17 transition `+1` | **CONFIRMED** | The C3 and C2 traces respectively lose the current legality relay and the current window fill if the transition is omitted. |
| R8, D14 recurrence | **CONFIRMED** | The maximum future Defender-placement count obeys the displayed bases and `1+max` recurrence exactly. |
| R9, legality coefficient eight | **CONFIRMED** | Radius-eight relays occur with equality. This is a limit of distance-only chain accounting. |
| R10, D15 ranks and deadlines | **CONFIRMED** | AND `+1`, OR `+0`, maximum over roles, rank zero at the check, leaf-entry protection, and the OR-COMPLETION role all survive. The last two have valid false-certificate gadgets. |
| R11, LOSS deadline | **CONFIRMED** | Separated count-four gadgets attain `leaf-ply+b+2` for both budgets. |
| R12, D16 recurrence | **CONFIRMED** | It is the exact path maximum for the stated exposure stop. |
| R13, T5 `B<=3` cutoff and `r3` | **REPAIR** | The displayed local geometry attains radii four and three, but the `B`/`E^D` values are not realized by a complete certificate and one needed exclusion of other A-touched incident windows is unstated. |
| R14, L10 first-three cutoff | **CONFIRMED** | The fourth future threat-creating placement is legal, is outside `r3`, and is in no currently A-touched window. |
| R15, nonempty AND fallback as an absolute pin | **REFUTED** | A zero-child AND node is maximal and violates the independent typed-maximal-node grammar; D14 and D16 also leave `max_C` undefined. Deleting only nonemptiness does not admit the alleged false certificate. |
| R16, forced-hit debit | **CONFIRMED** | `OPEN` is the correct boundary. No proof supplied here supports an `F+H_W` debit. |

### 1.2 L13-plus: the `3/5` theorem

The singleton case is valid, but the missing inference should be made
explicit. If `{a}` is a member and the subfamily `H` of members missing `a`
had `tau(H)<=1`, then `{a}` together with an at-most-one-point transversal of `H`
would hit the whole family with at most two points. Thus `tau(H)>1`; the
`b=1` selection takes at most three members of `H`, and adding `{a}` gives at
most four, hence at most five.

Assume all members have size two and let `G` be inclusion-minimal with
`tau(G)>2`. If `|G|=6`, a maximal disjoint subfamily has exactly two members

```text
E1={a,b}, E2={c,d}.
```

It cannot have one member, because that member would hit `G`, and it cannot
have three, because those three would be a proper obstruction already. Each
of the four cross-pairs needs a member of `G` disjoint from it. Minimality at
size six forces those four members to be distinct from one another and from
`E1,E2`. A member cannot miss two cross-pairs, because reusing it in the L13
selection would give a proper obstruction of size at most five. The member
missing `{a,c}` must therefore meet the other three cross-pairs; being a
two-set disjoint from `a,c`, it is exactly `{b,d}`. Cycling gives all six
edges of `K4`.

That `K4` cannot be a Hexo threat-empty family. Four pairwise axis-collinear
cells either lie on one axis line or do not exist. In the latter case,
translate one vertex to zero and put the other three on the three axes as
`(u,0)`, `(0,v)`, and `(w,-w)`. Pairwise alignment forces
`u=v`, `w=u`, and `w=-v`, a contradiction for distinct vertices. If three
vertices share a line, an off-line fourth has only two nonparallel axes that
meet that line and cannot align with all three. If all four share a line,
order them `p1<p2<p3<p4`. A consecutive length-six window containing
`p1,p3` contains the intervening empty `p2`; if their distance exceeds five,
there is no such window at all. In neither case can `{p1,p3}` be a two-cell
threat empty set.

Independent enumeration found exactly three threat windows in the triangle
position and exactly five in the `C5` position, not merely three or five
distinct empty sets. The `C5` position has 20 Attacker stones, five Defender
blockers, no complete window, no Defender `own_win_now`, transversal number
three, and minimum obstruction size five. All three axes and all incident
windows were included. L13-plus is install-ready.

The normative cap edits are:

- in D9, replace `|T| <= 6 for b = 2` by `|T| <= 5 for b = 2`;
- replace normative L13 by the L13-plus proof in the tightness report; and
- in normative section 9, replace `3 ... and 6` by `3 ... and 5`.

### 1.3 Exact rank trace and the two overclaimed sharpness sentences

For the rank-two trace, `x0` and `f0` are legal at the first Defender edge.
The target `y` is initially ghost-illegal because it is distance 16 from
both shared supports. After the first edge, real `y` is legal through `x0`,
ghost `y` remains illegal, and `f1` is legal through `f0`. Starting with
Defender budget two gives exact ranks `2,1,0` across the two AND edges and
the deadline; intervening OR edges add zero. The placement `a` is Z4-legal
through `z`, and `y` is then Z4-legal through `a`. With only the displayed
stones, no transition completes a window.

For arbitrary `r`, the `x_i` and `f_i` relays are all exactly distance eight,
and `d(x0,y)=8(r-1)`. Starting at budget two for even `r` and budget one for
odd `r` makes the `r`th Defender placement end a turn. Shared Attacker
fillers can be kept remote from `y` and chosen without making six in line.
This proves sharpness of the exact-rank L9-prime chain. It does not provide a
typed terminal certificate with `B=r`; the report correctly leaves the
uniform wrapper open.

The normative sentence after L9-prime should be replaced exactly by:

> The distance-chain inequality is sharp for an exact role rank `r`: a
> ghost-legal seed may be followed by `r-1` successive distance-eight
> Defender placements, with the protected target last. This does not
> establish sharpness of the uniform `B`-only wrapper.

The fixed-window virgin trace is also arithmetically correct. For
`k=E^D-6`, its first relay is distance `8k` from the target, every relay step
is eight, and the last relay is distance eight from the first target fill.
Parity permits the final fill to end a Defender turn for every `E^D>=6`.
It does not pin the full verifier union: the legal seed lies in 18 incident
windows, and any incident window that is still virgin with exposure at least
six may select it at distance zero. The
normative sentence after L12 should be replaced exactly by:

> For one fixed window, the causal counting inequality is attained at
> `E^D=7`: a legal seed at distance eight may be followed by the first
> `W`-fill and the five remaining fills. This does not pin the full union
> `Z_virgin`.

Normative section 12, item 2, should replace `including the sharp virgin
radius` by `including the fixed-window virgin radius`; sharpness of the full
union remains open.

### 1.4 Touched and T5 trace qualifications

The four touched-window fillers are legal, the ghost never completes a
window, and the real count is exactly `2,3,4,5,6`. This confirms that the raw
inequality and turn parity contain no numerical slack. The report does not,
however, give a finite terminal subtree from which D16 derives exposure four,
nor all searched sets and other zone terms.

Change frontier row R3's result cell to `fixed-window equality attained; full
weakened-L12 pin OPEN`, and change the section 3.2 heading to:

> `### 3.2 Touched equality -- fixed-window arithmetic attained; full pin OPEN`

Replace the paragraph beginning `At the root` by:

> At the root the displayed fixed-window count is `2+4=6`. If one stipulates
> that the remaining target-window exposure falls to three after the first
> ghost edge, the target-window arithmetic does not later recapture `w_1`.
> No recurrence-derived terminal certificate is supplied, so this display
> makes no claim that every other completion or obligation zone also omits
> the real fills. It shows only that neither turn parity nor the raw exposure
> arithmetic supports replacing `>=6` by `>6`; it does not by itself falsify
> L12.

Replace the paragraph beginning `For budget one` by:

> For budget one, the displayed real sequence attains the same fixed-window
> equality and completion. This remains an arithmetic trace, not a full
> L12 counterexample. A full pin requires a complete D9 certificate with
> recurrence-derived exposure, a terminal subtree, searched sets, and all
> other zone clauses checked.

The T5 `B=4` cell is legal and is distance four from the displayed stones;
the `B=3` endpoint is distance three. To make the local exclusion complete,
insert after the first display in section 8.1:

> Take no Attacker stone in any of the 18 windows through `(-5,0)` and no
> other current stone within distance three of that cell; put any remaining
> certificate data remotely.

Change frontier row R13's result cell to `local radius arithmetic attained;
full T5 pin OPEN`, and change the section 8.1 heading to:

> `### 8.1 T5 cutoff and radius -- local arithmetic attained; full pin OPEN`

Replace `Thus the same static set cannot extend to B=4` by:

> Thus the local static-cover arithmetic cannot extend unchanged to `B=4`.
> A full T5 pin additionally requires a D9 certificate realizing the stated
> `B` and `E^D` labels; the displayed coordinates alone do not supply it.

Also replace `lies in Z_touch because 2+4=6` by:

> satisfies the local `Z_touch` arithmetic under the stipulated label
> `E^D(W)=4`, because `2+4=6`

Replace the paragraph beginning `Radius three itself is attained` by:

> At the `B=3` endpoint, three Defender stones at offsets three, four, and
> five put the offset-zero empty at distance exactly three while satisfying
> the local count equality. Choose no Attacker stone in any window through
> the offset-zero cell and put required Attacker/root data remotely, so the
> A-touched half of the static union does not select it. Radius two misses it.
> This also establishes local arithmetic only until a complete certificate
> realizes the stated `B` and exposure labels.

The L10 example needs no repair. The target `(5,5)` is distance six from its
nearest current stone. It shares no axis window with the current `3 by 3`
Attacker block, and the only collinear current anchor is six cells away. The
first three placements each create a vertical count-four threat; the fourth
creates the horizontal count-four threat supported entirely by those three
future stones.

### 1.5 Absolute-pin audit

The T6 `not own_win_now` game counterexample is correct. At the root the
complete Attacker threat family is `{k}`, `K_1={k}`, and the searched `k`
edge is legal and nonterminal. The designated Attacker moves `p,d` are legal
and nonterminal. At the resulting leaf the three named pairs are actual
threats with transversal number three, and Defender `own_win_now` is false.
The omitted real move `d` is legal and completes `U` immediately.

It is not, however, a weakened certificate satisfying every other D9 clause.
D9 separately retains the internal-AND `not own_win_now` diagnostic, and the
root fails it. Delete the sentence `The only failed T6 clause is the deleted
root not own_win_now premise.` Replace the last paragraph of tightness
section 5.2 by:

> This position proves that a kernel verifier cannot accept a node with
> `own_win_now`. It is an absolute counterexample only to a combined
> weakening that deletes the explicit T6 premise and ceases enforcement of
> D9's retained internal-AND diagnostic. Deleting the T6 premise alone still
> leaves the diagnostic rejecting this root. A replacement kernel may instead
> search every immediate Defender completion or reinstate the ordinary
> completion zone.

The frontier table and final limit map must describe R5b as absolute only for
removal of the predicate from the kernel verifier as a whole, not for deletion
of the T6 clause in isolation. Change its frontier result cell to `PINNED
absolutely for combined predicate enforcement; not a single-clause pin`, and
change the section 5.2 heading to:

> `### 5.2 Kernel-node not-own-win requirement -- combined enforcement PINNED`

Use this final-map row:

> `| R5b | **Absolute for combined predicate enforcement** | If neither the T6 premise nor the retained D9 diagnostic is enforced, the kernel misses an immediate Defender win |`

The leaf-entry gadget also survives. Ghost `s` is legal and nonterminal. At
the `b=1` leaf its two named threats have transversal number two and Defender
has no immediate win. In the real line, `u` and then `v` are legal,
nonterminal, and kill both threats. The ordinary completion terms are empty
at the root because there is no current Defender stone and the exposure is
only two.

The OR-COMPLETION gadget survives. Ghost `s` is a legal nonterminal reply and
ghost `c` completes the named window. Real Defender can instead occupy `c`;
after that move every current count-four-or-more Attacker-alive window is
dead. Omitting the designated OR-COMPLETION role alone admits the false line.

The LOSS deadline is attained. Use copies of the count-four gadget translated
by `(20j,0)`, for `0<=j<=b`. The `b+1` empty pairs are disjoint,
`tau=b+1`, and Defender has no immediate win. Defender kills `b` gadgets;
the survivor remains at count four. The next Attacker placement makes count
five and the following placement first completes. Resolution is exactly
`leaf-ply+b+2`.

R6 needs a scope repair. D9 separately states both `tau(T)>b` and the
universal requirement that every complete nonterminal remainder leave a
named witness untouched. The equality examples violate both. Replace the
first paragraph of section 4.4 by:

> The inequality `tau(T)>b` is the finite characterization of D9's
> universal survivor clause. Equality is an absolute counterexample to
> weakening the LOSS survivor contract itself; it is not a certificate
> satisfying the unchanged universal clause. Deleting only the numeric test
> is harmless if the universal clause is still verified.

Replace the concluding sentences `Thus accepting tau=b makes the D9 leaf
contract false. This is an absolute pin.` by:

> Thus equality defeats the combined LOSS survivor contract. It is not an
> absolute pin of the numeric test alone while the universal-survivor clause
> remains enforced.

The final limit map must describe R6 as absolute only for the combined LOSS
contract, not for the numeric test in isolation. Change the section 4.4
heading to:

> `### 4.4 Combined LOSS survivor contract -- PINNED`

Use this final-map row:

> `| R6 | **Absolute for the combined LOSS survivor contract** | Equality permits a complete remainder hitting every witness; deleting only the redundant numeric test does not |`

R15 is refuted as an absolute pin. A zero-child AND is a maximal node, but D9
requires every maximal node to be a typed WIN, LOSS, or OR-COMPLETION leaf.
It also gives no value to the D14 and D16 maxima over children. Replace the
body of section 7.6 by:

> Step A2/A3 needs a legal searched reply to consume a ghost Defender edge
> when the real reply is occupied or dismissed. A zero-child AND supplies no
> such filler. It is also a maximal node without a typed terminal label, and
> the D14/D16 maxima over its children are undefined. Thus nonemptiness is
> exact as a syntactic well-formedness and coupling-filler requirement.
> Deleting it alone does not admit a false certificate. R15 is a
> relative/syntactic pin, not an absolute pin.

The frontier table and final limit map must change R15 from `Absolute` to
`Relative/syntactic`. Change the section 7.6 heading to:

> `### 7.6 Nonempty searched fallback -- PINNED syntactically/relatively`

Use this final-map row:

> `| R15 | Relative/syntactic | A zero-child AND has no coupling filler and is not defined by the current D9/D14/D16 grammar |`

### 1.6 Other tightness rows

The remaining relative rows survive as stated. The D17 C3 trace has child
rank one and parent distance eight, so a child-only radius zero loses the
current relay. The C2 trace has two old Defender stones, the omitted current
fill, and three later fills; `2+3` misses the completion while `2+1+3`
catches it. D14 and D16 are exact maxima over finite paths and obey their
displayed recurrences. The kernel residual threshold and the legality factor
are attained. None of these relative results excludes a verifier using more
state or a different proof.

### 1.7 Tightness checker audit

The checker is useful but its advertised coverage is false. Replace report
lines 17-20 by:

> `scripts/_tightness_check.py` verifies the triangle and five-cycle threat
> families, selected facts of the T6 counterexample and deadline gadgets,
> and isolated distance/count equalities. The remaining prose arguments are
> not machine-checked by this script.

Specific findings are:

- `exact_family` converts the actual threats to a set of empty sets. An extra
  distinct threat window with a duplicate empty set would pass. The actual
  constructions independently have exactly three and five windows. Replace
  the construction of `actual` by:

  ```python
  actual_threats = threats(attacker, defender)
  assert len(actual_threats) == len(expected)
  actual = {empty for _, _, _, empty in actual_threats}
  assert actual == set(expected), (actual, set(expected))
  ```
  In the five-cycle block also add `assert len(attacker) == 20` before the
  family check.
- The T6 block checks the root family and kernel, one Defender completion,
  leaf-family inclusion, `tau`, and leaf `own_win_now`. It does not check all
  exact successor phases, nonterminality of `k,p,d`, or the full D9 leaf
  contract.
- The deadline block does not model either certificate. In particular, its
  LOSS half checks the threat family with no ghost `s` on the board.
- The rank block checks the displayed distances only. It does not check
  ranks, turn boundaries, Z4, terminality, arbitrary `r`, the virgin support
  and parity construction, the touched fillers, or complete B/D16 labels.
- There are no assertions for R6, R11, R15, L10, or the D17 coordinate
  traces.

The checker must not be run with optimized Python. Add after its imports:

```python
if not __debug__:
    raise RuntimeError("this verifier requires assertions; rerun without -O")
```

## 2. Pairing report

### 2.1 Verdict table

| Item | Verdict | Finding |
|---|---|---|
| Equality forcing | **CONFIRMED** | All three incidence inequalities and all equality conclusions are valid after passage to a locally injective period sublattice. |
| Line rigidity | **REPAIR** | `x_s=x_(s+6)` is correct. Periodicity is used to force exact coverage; the recurrence is pointwise once exact coverage is known. |
| Search-space completeness | **CONFIRMED** | The six phase variables cover all pairings periodic under the displayed `Lambda`, not only an unjustified subclass. |
| Explicit matching | **CONFIRMED** | The 12 endpoints are the 12 quotient cells exactly once. |
| Window coverage | **CONFIRMED** | One phase occurs on every physical line of every axis; every length-seven window contains exactly one pair. |
| Period-index minimality | **CONFIRMED** | Every axis-step order is divisible by six, and an order-six quotient cannot give order six to all three axis steps. |
| Search/checker | **REPAIR** | The two enumerators exhaust the stated phase model, but the exact reported 419-state count is not asserted and both share that same model. |

### 2.2 Equality and rigidity

After passing to a period sublattice with no nonzero vector of diameter at
most six, quotient endpoints and the windows containing a relevant pair are
distinct. There are `3N` start/axis window orbits. Coverage gives

```text
3N <= sum_e (7-delta_e).
```

Every relevant pair has `delta_e>=1`, so the sum is at most `6P`. The
matching gives `2P<=N`, hence `6P<=3N`. Equality throughout forces
`P=N/2`, every pair to be a unit pair, every cell to be matched, and the
total incidence count to be exactly one per window. Axiswise there are `N`
window orbits and a unit pair covers six of them, so there are `N/6` pair
orbits on each axis.

On a physical line, a length-seven window has the six internal unit-edge
starts `s,...,s+5`; exact coverage gives the displayed six-term equation.
Subtracting consecutive equations gives `x_s=x_(s+6)`. One of the six
residues is selected, so the least positive period is six.

Replace report lines 39-41 by:

> Obtaining exact coverage above uses periodicity; once exact coverage is
> known, the recurrence is pointwise. The Folner-density argument alone does
> not exclude zero-density defects.

### 2.3 Construction, complete search space, and hand coverage

For `Lambda=<(2,2),(0,6)>`, the intersections with each axis subgroup are
exactly the six-step subgroups. Physical line orbits are `r mod 2` for the
horizontal axis, `q mod 2` for the vertical axis, and `q+r mod 2` for the
diagonal axis. Thus there are exactly two line orbits on each of three axes.
Lambda-periodicity and rigidity give one of six phases on each orbit: the
script's six variables cover every Lambda-periodic covering pairing. The
endpoint columns impose exactly the remaining matching condition.

The displayed Lambda is not itself locally injective: `(2,2)` has hex
distance four. This is harmless. Local injectivity is used on a deeper
sublattice to derive unit pairs and exact physical coverage; those are
properties of the pairing itself and then descend to the six-variable
Lambda encoding.

The quotient map has kernel Lambda. Its six representative pairs use all 12
elements of `Z_2 x Z_6` exactly once, proving the matching independently of
the search. Coverage can also be checked by hand:

- horizontal starts on the two representative line orbits are `(6n,0)` and
  `(6n,1)`;
- vertical starts are `(0,3+6n)` and `(1,3+6n)`; and
- on the odd- and even-sum diagonal representatives, starts have
  `q=1 mod 6`.

Lambda translates cover every line. Any six consecutive internal edge
starts contain one selected residue, so every length-seven window contains
exactly one pair.

### 2.4 Minimality and checker audit

For an original period lattice, let `o` be the order of an axis step in the
quotient. Translation by `o` steps is a period of that line indicator. Its
least period is six, so `6|o`; Lagrange gives `o|N`, hence `6|N`. If
`N<12`, then `N=6`. The abelian quotient of order six is cyclic. Both first
axis steps must be `g` or `-g`; their difference is zero or `+/-2g`, of order
at most three, while the third axis step must have order six. This is the
required contradiction.

The fresh run found 120 Algorithm-X solutions in 419 recursive calls and 120
solutions in the direct `6^6` loop. The quotient histograms were `{1:12}` and
`{1:36}`. These checks are nonvacuous. They both assume the same six-phase
model, so they do not machine-prove the prose completeness reduction; section
2.3 supplies that proof.

The report states the exact recursive-state count. Add immediately after
`assert solutions == 120`:

```python
assert states == 419
```

Also add the `__debug__` guard stated in section 1.7. The finite patch is a
redundant periodicity check, not additional evidence for the completeness of
the phase model.

## 3. ES report

### 3.1 Verdict table

| Item | Verdict | Finding |
|---|---|---|
| Exact `Q(sqrt(3))` arithmetic | **CONFIRMED** | The weights, profile formula, sign comparison, and strict comparisons are exact. |
| Theorem 1, all greedy ties | **CONFIRMED** | Every positive maximum is branched; the fixed Attacker continuation wins on every reachable greedy state. |
| State deduplication | **CONFIRMED** | Only identical coloured boards at the same phase are merged; future legality, danger, and the fixed continuation are Markovian. |
| Proposition 1, reachable enlargement | **CONFIRMED** | The 39 placements are legal and nonterminal, the padding is dead and separated, and the full enlarged greedy tree is rerun. |
| Lemma 1, clean escape | **CONFIRMED** | The `h` bounds exclude old stones on all three axes and the two new cells share no window. |
| Corollary 2 and repeated source | **REPAIR** | The nonsummable fresh-label source is correct; `disjoint stars` should not suggest disjoint residual cell supports. |
| Theorem 2 | **CONFIRMED** | Initial alive/dead/virgin windows are exhaustive, and virgin completion needs six future Attacker placements. |
| Theorem 2 sharpness trace | **CONFIRMED** | It starts at `Phi=0`, follows the specified filler exactly, and wins only on Attacker placement six. |
| Lemma 2 | **CONFIRMED** | Both danger inequalities and the factor `3/2` remain valid with overlapping windows. |
| Theorem 3 | **CONFIRMED** | The three geometric cases are exhaustive; both applications of Lemma 2 are legitimate. |
| Proposition 2 | **CONFIRMED** | The internal-window count and the transition from (25) to (26) are correct. |
| Proposition 3 | **CONFIRMED** | The account chain (28) follows directly and contradicts finite virgin mass. |
| Theorem 4 | **CONFIRMED** | Finite branching, radius `32h`, bound (30), and the Konig compactness argument are valid. |
| Checker claims | **REPAIR** | The checker is exhaustive for greedy maxima along the fixed Attacker continuation. Some broader wording and several literal subclaims are not asserted. |

### 3.2 Greedy counterexample and checker quantifiers

The initial cell pair kills exactly five of the 18 windows through the
Attacker stone, leaving profile `(13,0,0,0,0)` and
`Phi=13 sqrt(3)/27<1`. The exact checker branches every maximizer at every
Defender placement. All maxima are positive, so omitted zero-danger legal
cells cannot tie. Deduplication is sound: two identical coloured boards at
the same phase have the same legal moves, alive labels, exact dangers, win
predicates, and remaining fixed Attacker schedule. A history-dependent tie
rule gains nothing from retaining two copies of the same state, because the
merged state again branches over every exact maximum.

Before `D3.1`, independent inspection gives incident Attacker counts
`(1,2,3,4,4)` at `(-6,0)` and `(4,5)` at `(-1,0)` on all 124 states. Thus
the two dangers are exactly `21+4sqrt(3)` and `9+9sqrt(3)`. After the forced
first choice, `(-1,0)` has only its count-five term, `9sqrt(3)`, and each
stated second maximum is strictly larger. The final fixed placement completes
the target on all 124 states.

Replace report lines 49-54's final sentence by:

> The enumeration has no randomness, beam, or depth cutoff and omits no
> exact greedy maximizer along the fixed Attacker continuation.

The original phrase `no omitted move` is broader than the quantified search.

### 3.3 Reachability and clean escape

The expanded history has the correct side order and ends at Defender
FirstStone with 20 Attacker and 19 Defender stones. Every placement is
radius-eight legal and every prefix is nonterminal. Every window through an
inner-ball Attacker stone meets the Defender ring. The padding and every
tactical cell are at distance at least six, so a length-six window of diameter
five cannot cross components. The checker calls the same exhaustive
`run_trace` on the expanded position; it is not sampling branches.

For clean escape, `h(x)=M+8`. A Q- or R-window through `x` has minimum
`h>=M+3`, and a QR-window has constant `M+8`. At `y`, the corresponding
bounds are `M+11` and `M+16`; these also exclude `x`. The displacement
`(4,4)` has hex distance eight but is on no window axis, so the two cells
share no window. Exactly 36 fresh count-one window labels are born, of mass
`36 lambda^-5=4/sqrt(3)`.

Replace report lines 311-316 by:

> Repeating Lemma 1 at every continuing Attacker turn creates 36 fresh,
> distinct count-one window labels and a source term `4/sqrt(3)` on every
> such turn. Labels born on different turns are distinct, although their
> residual cell supports need not be disjoint. No clean-escape placement
> promotes an earlier label, because every window through that placement was
> stone-free immediately beforehand. In the conservative blanket game that
> ignores Defender completions, the cumulative birth sum therefore diverges
> while every such label either remains at count one or is killed.

The conclusion remains a no-go result for renewal proofs, not an Attacker win
against arbitrary defense.

### 3.4 Finite-horizon proofs

Theorem 2's three classes are exhaustive: an initial window is Attacker-alive,
contains a Defender stone, or is stone-free. Fixed-family greedy protects the
first forever, permanence protects the second, and the third cannot acquire
six Attacker stones during the next five Attacker placements. The sharpness
display starts with no Attacker-alive window and hence `Phi=0`. Each Defender
move is exactly the maximum-`q`, then maximum-`r`, positive-Q filler. No
earlier placement wins; the sixth Attacker placement first completes the
vertical target. This does not contradict the five-placement theorem.

For Lemma 2, after the arbitrary Defender move `X'<=X`. The greedy reduction
`delta` is the pre-placement maximum; deletion gives `S<=delta`. Also
`S<=X'-delta`, because a cell danger is a subsum of the surviving labelled
potential terms. Geometric overlap does not duplicate a label in that
subsum. The Attacker pair adds at most
`(lambda-1)(1+lambda)S=2S`. Splitting at `delta=X'/2` gives

```text
X'-delta + 2 min(delta,X'-delta) <= 3X'/2 <= 3X/2.
```

The prefix claim uses the smaller first-placement increase, so terminal
checking is valid.

For Theorem 3, a newly activated window completing on future Attacker
placement six must contain all six future Attacker cells, including the first
pair `x,y`. If they share no window, no virgin target exists. At axis distance
two through five, `x+v` lies in every common length-six window. In the adjacent
case, `x-v` hits the first four of the five common windows. Only `W*` can
remain; before pair three it has at most four Attacker stones and at least two
legal empties. The first arbitrary placement on each of the next two Defender
turns respectively kills those targets, and the second placement is
F-greedy. Lemma 2 therefore applies twice exactly as claimed. The thresholds
are `2/3` and `(2/3)^2=4/9`.

### 3.5 Counting and compactness proofs

For Proposition 2, every axis contributes exactly
`|B_R|-5(2R+1)` internal length-six windows when `R>=5`. A pair covers at
most five windows, so `C` covered windows require at least `2C/5` matched
cell incidences in the ball. Comparing with the at-most `|B_R|` matched cells
gives

```text
|B_R| <= 60R+30+2K,
```

which contradicts quadratic growth. A finite support meets at most `18|S|`
windows, so the finite-exception conclusion follows.

Proposition 3's chain
`1<=w6<=3w4<=9w2<=27w0` is exact. Nonnegativity then forces at least `1/27`
virgin weight on infinitely many windows.

For Theorem 4, at most `4h` placements occur in `h` rounds. Induction gives
distance at most `8k` from the original support at placement `k`, hence
radius `32h`. The ball formula gives exactly
`3072h^2+96h+1` per original support cell. The truncated game tree is finite.
Add an artificial level-zero root to the tree of surviving finite policy
tables. Restriction maps every surviving level-`h` policy to a surviving
level-`h-1` policy. Levels and branching are finite and every level is
nonempty, so Konig's lemma gives a consistent infinite chain. Its union is a
single forever-surviving Defender strategy. The converse finite-minimax
statement follows.

### 3.6 ES checker repairs

The exact surd comparison is correct for every sign pattern. Irrationality
rules out equality when the two nonzero coefficients have opposite signs.
The reachable-position run is a full rerun, not a sample. The following
repairs align literal checker coverage with the prose:

1. Add this module-level helper before `main`:

```python
def incident_counts(pos: Position, x: Coord) -> tuple[int, ...]:
    return tuple(sorted(s for _w, s, empties in alive(pos) if x in empties))
```

Immediately after `pre_d3` is assigned inside `main`, add:

```python
    assert Counter(incident_counts(p, (-6, 0)) for p in pre_d3) == Counter(
        {(1, 2, 3, 4, 4): 124}
    )
    assert Counter(incident_counts(p, (-1, 0)) for p in pre_d3) == Counter(
        {(4, 5): 124}
    )
```

Immediately after `post_d31` is assigned, add:

```python
    assert Counter(incident_counts(p, (-1, 0)) for p in post_d31) == Counter(
        {(5,): 124}
    )
```

2. The current aggregate comparisons do not literally assert a translated
   state-set bijection, although the separation proof establishes it and the
   expanded run independently verifies every branch. Add after both traces
   are built:

   ```python
       padding_a = expanded.A - shifted_set(compact.A, ENGINE_SHIFT)
       padding_d = expanded.D - shifted_set(compact.D, ENGINE_SHIFT)

       for (c_name, c_states), (e_name, e_states) in zip(
           compact_trace.stages, expanded_trace.stages, strict=True
       ):
           assert c_name == e_name
           translated_states = {
               Position(
                   shifted_set(p.A, ENGINE_SHIFT) | padding_a,
                   shifted_set(p.D, ENGINE_SHIFT) | padding_d,
               )
               for p in c_states
           }
           assert set(e_states) == translated_states
   ```

3. Add the `__debug__` guard from section 1.7. Under `python -O`, the current
   assertions disappear while the script still prints `VERIFIED`.

## 4. Reproduction record and final rulings

The assertion-enabled runs reproduced these principal outputs:

```text
_tightness_check.py: all tightness checks passed
_pairing7_search.py: 120 solutions; 419 recursive states; quotient {1:12}/{1:36}
_es_global_check.py: GREEDY-REFUTATION VERIFIED; 39-placement reachable history
```

This review created only `docs/_REVIEW_TPE_ROUND1.md`; it did not edit any
pre-existing document or script.

`docs/_TIGHTNESS_FRONTIER_REPORT.md (+ scripts/_tightness_check.py)`: **INSTALLABLE-WITH-REPAIRS**.

`docs/_OPEN_PAIRING7_REPORT.md (+ scripts/_pairing7_search.py)`: **INSTALLABLE-WITH-REPAIRS**.

`docs/_OPEN_ES_GLOBAL_REPORT.md (+ scripts/_es_global_check.py)`: **INSTALLABLE-WITH-REPAIRS**.

The tightness report has the only refuted classification: R15 is not an
absolute false-certificate pin. R5b and R6 require combined-clause scope
repairs before they may be called absolute. L13-plus itself is sound. The
pairing existence/minimality theorem and the ES `GREEDY-REFUTED` verdict
survive.
