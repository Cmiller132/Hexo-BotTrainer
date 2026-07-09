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

Later gate results, audit evidence, cost rankings, and the final recommendation
will be filled as each ordered gate turns green.
