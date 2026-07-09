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

### D4: CPU model import otherwise imports Triton indirectly

Production `model.py` unconditionally attempts to import
`torch.nn.attention.flex_attention` even when every flex and Triton environment
gate is off. On the installed Windows Torch build that import loads the Triton
package before falling back with a no-CUDA warning, conflicting with Phase A's
"no Triton imports" rule. New CPU-only code therefore disables every backend
gate and places a `None` sentinel for that optional module before importing the
production model; the guarded import takes its existing no-flex fallback. G6
asserts that no `triton` module entered `sys.modules`. The same guard is used by
the G7 CPU audit.

### D5: concurrent feature-v2 work appeared after Phase A started

The initial ground-truth tree and live checkpoint use the specified 25-plane
input (`13*triv + 4*axis`). During execution, unrelated unstaged edits appeared
in `constants.py`, `equivariant.py`, `features.py`, and the Rust constants that
add an opt-in `HEXFIELD_EQ_FEATURE_VERSION=2` 46-plane map. Phase A neither
authored nor modified those files. Its stem proof remains intentionally scoped
to feature version 1 because the work order, CONTEXT, and audited checkpoint all
require 25 planes; every G3–G7 process asserts version 1. Generalizing the typed
stem to version 2 would be a separate change to a moving, uncommitted design and
would violate this phase's strict gate contract.

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

### G5 — nonlinearity legality: PASS

The positive control applies GELU under every element of every required
single-type permutation representation. The negative control independently
constructs the determinant/sign representation, verifies its homomorphism law
for all 144 products, and measures the reflection-equivariance violation of
GELU on a fixed fp64 vector.

Every quotient action commuted within fp64 tolerance. Every reflection in the
valid sign representation violated GELU commutation by more than 1.0, while
rotations (sign `+1`) remained exact. The combined G4/G5 file now reports
**5 passed** on Windows and **5 passed** in WSL.

### G6 — typed toy network: PASS

The toy architecture uses both required signatures (widths 51 and 96), a typed
25-plane Reynolds stem, two production-form typed conv residual blocks, a
regular `reg:4` attention interior with 3 coset heads of dimension 16, joint
pair/head bias, a regular-interior sigmoid-gated SUM register refresh, typed
MLP, and invariant policy/value reads. The end-to-end gate covers five seeded
legal connected prefixes expanded by the real Python oracle and all 12 D6
transforms. A pure `reg:8` construction separately parameter-matches production
stem, two `ConvBlock`s, and one materialized `AttnBlock` at width 96.

Both signatures passed 120 end-to-end transformed forwards (2 signatures × 5
positions × 12 elements): policy rows permuted and scalar values stayed
invariant at `atol=1e-9`. The parameter-matched pure-regular stem/C/C/A path
agreed with production primitives at fp64 tolerances of `1e-12` through the
stem, `1e-11` through conv blocks, and `1e-10` through attention. The test
asserts before and after execution that no `triton` module is loaded.
`tests/test_hexfield_eq_reps_toynet.py`: **2 passed** on Windows and **2 passed**
in WSL.

Later gate results, audit evidence, cost rankings, and the final recommendation
will be filled as each ordered gate turns green.
