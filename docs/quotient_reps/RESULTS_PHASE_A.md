# Phase A results: quotient representations

Status: in progress. Measurements in this document are CPU-only.

## Contract discrepancies found during execution

### D1: the stated rot180 identification is ambiguous

The Phase-A spec identifies rot180 as the element of
`K = {0,3,7,10}` that sends `(1,0)` to `(-1,0)`. Direct evaluation through
`geometry.apply_d6` shows that both `g3` (rot180) and `g10` (a reflection)
satisfy that one-vector condition. The unambiguous correction used by G1 is
the unique orientation-preserving element of K that reverses both axial basis
vectors: `g3: (1,0) -> (-1,0), (0,1) -> (0,-1)`. The required reflection is
uniquely `sigma = g7`, the reflection in K that fixes `(1,0)`.

### D2: the two mandated channel layouts conflict

Constraint C4 inherits production's slot-major layout, while spec section 2
states instance-major/slot-minor layout. With multiplicity greater than one
these are related by a perfect shuffle but are not the same representation
matrix in raw channel order. Therefore arbitrary pure-regular typed weights
cannot both use the section-2 order and equal production weights elementwise
as G3 requires. G3 and Phase-B D8 make the correction unambiguous: inside each
type block Phase A uses the production-compatible order
`channel = type_offset + slot * multiplicity + instance`. This is frozen by
tests and will be reflected in the derivation.

### D3: the literal "ANY environment" import requirement exceeds its imports

The spec requires `reps.py` to import `geometry` and `constants.DIRECTIONS` but
also says it must import under any `HEXFIELD_EQ_*` environment. Those mandated
imports execute the existing `constants.py` validation, which intentionally
rejects invalid configurations before `reps.py` runs. The achievable contract
is that `reps.py` introduces no channel/group-shape dependency of its own and
imports in every environment in which the mandated geometry/constants modules
already import.

## Gate results

### G1 — group foundation parity: PASS

`tests/test_hexfield_eq_reps_group.py` contains four tests. They establish exact
equality of all seven fields returned by production `build_group`, derive
`sigma=g7` and `rot180=g3`, freeze canonical left-coset ordering, and check the
permutation/homomorphism laws for all five types and all 144 group-element
pairs. Results: **4 passed** on Windows Python and **4 passed** in the specified
WSL venv.

### G2 — Hom-space dimensions: PASS

Rows are input types and columns are output types. The fixed order is
`reg, mirror, point, axis, triv`.

| in \ out | reg | mirror | point | axis | triv |
|---|---:|---:|---:|---:|---:|
| reg | 12 | 6 | 6 | 3 | 1 |
| mirror | 6 | 4 | 3 | 2 | 1 |
| point | 6 | 3 | 6 | 3 | 1 |
| axis | 3 | 2 | 3 | 2 | 1 |
| triv | 1 | 1 | 1 | 1 | 1 |

The corresponding conv-tap dimensions (reported as additional evidence) are:

| in \ out | reg | mirror | point | axis | triv |
|---|---:|---:|---:|---:|---:|
| reg | 84 | 42 | 42 | 21 | 7 |
| mirror | 42 | 24 | 21 | 12 | 5 |
| point | 42 | 21 | 24 | 12 | 4 |
| axis | 21 | 12 | 12 | 7 | 3 |
| triv | 7 | 5 | 4 | 3 | 2 |

For all 25 linear pairs, diagonal pair-orbit counts, directly enumerated
double-coset counts, and fp64 SVD ranks of the Reynolds projectors agree. For
all 25 conv pairs, triple-orbit counts agree with independent fp64 projector
ranks; the required `reg -> reg` anchor is 84. Each projector is also checked
symmetric and idempotent at `atol=1e-12`. The table was independently rebuilt
from `apply_d6` by a second analysis and matched entry-for-entry. Results:
`tests/test_hexfield_eq_reps_homdims.py` **4 passed** on Windows and **4 passed**
in WSL.

### G3 — production machinery reproduction: PASS

The pure-regular specialization uses an explicit transversal bijection:
production `wb[s]` maps to the orbit containing `(out=e, in=s)`, and production
`w_base[t,s]` maps to the orbit containing `(tap=t, out=e, in=s)`. This gives
12 and 84 distinct basis coefficients respectively. The stem comparison uses
the generated 25-plane input action and the typed-output Reynolds lift.

Random fp64 production parameters were reproduced elementwise: linear and
conv weights at `atol=0, rtol=0`, and the averaging stem lift at
`atol=1e-12, rtol=0`. The input representation matrices also match production
exactly. `tests/test_hexfield_eq_reps_parity.py`: **4 passed** on Windows and
**4 passed** in WSL.

### G4 — typed layer property tests: PASS

The seeded property suite samples 50 input/output signature pairs with each
type multiplicity drawn from 0 through 4. For every pair and every D6 element
it checks TypedLinear and TypedConv, including exact dense-weight orbit ties and
fp64 forwards. Conv forwards rebuild the support/neighbour table after
transforming the stone coordinates. A second 50-signature sweep checks the
full-fiber typed norm, per-instance LayerScale, typed group pool, ReLU, and
GELU.

All 50 seeded signature pairs passed all 12 group elements. Dense orbit ties
are asserted at `atol=0`; fp64 forward covariance is asserted at `1e-10`.
ReLU is bit-exact. PyTorch's vectorized CPU GELU produced a one-ulp
(`5.55e-17`) channel-order difference in one Windows case, so the mathematical
commutation check uses the fp64 averaging-grade tolerance; WSL also passes.
`tests/test_hexfield_eq_reps_typed_layers.py`: **3 passed** on each platform.

Later gate results, audit evidence, cost rankings, and the final recommendation
will be filled as each ordered gate turns green.
