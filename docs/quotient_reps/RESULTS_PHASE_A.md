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
and randomized per-instance bias invariance are asserted at `atol=0`; fp64
matmul forward covariance is asserted at `1e-10`, while typed norms, pooling,
and GELU use `1e-12`. ReLU is bit-exact. PyTorch's vectorized CPU GELU produced a one-ulp
(`5.55e-17`) channel-order difference in one Windows case, so the mathematical
commutation check uses the fp64 averaging-grade tolerance. The independent
adversarial review specifically confirmed that nonzero biases now participate
in the 50-signature sweeps.
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
stem, two `ConvBlock`s, `RegisterRefresh`, one materialized `AttnBlock`, final
norm, and policy/value readouts at width 96.

Both signatures passed 120 end-to-end transformed forwards (2 signatures × 5
positions × 12 elements): policy rows permuted and scalar values stayed
invariant at `atol=1e-9`. The parameter-matched pure-regular
stem/C/C/refresh/A/final/head path
agreed with production primitives at fp64 tolerances of `1e-12` through the
stem, `1e-11` through conv blocks, and `1e-10` through the downstream path.
The register comparison preserves production's explicit fp32 counting sum.
The test restores its optional FlexAttention import guard and asserts that it
imports no new `triton` module, so it does not poison later plain-pytest collection.
`tests/test_hexfield_eq_reps_toynet.py`: **2 passed** on Windows and **2 passed**
in WSL.

### G7 — real-checkpoint type audit: PASS

The required audit ran over 512 manifest-uniform real decision rows from the
checkpoint-aligned live sample set. It captured every post-block cell stream,
the corresponding register-token stream, and the pre/post-`ln_final` streams;
all projection energies were accumulated in fp64. The optional
`EquivLinear.wb` coefficient audit was also enabled as an extension.

Exact Windows command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='packages/hexfield_eq/python;packages/hexo_engine/python;packages/hexo_utils/python'
$report=Join-Path $env:TEMP 'hexgt_g7_epoch15_full.md'
python -B scripts/quotient_type_audit.py --checkpoint E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\checkpoints\epoch_000015.pt --shards E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\samples --positions 512 --batch-size 8 --threads 16 --weight-audit --verbose --output $report
```

Wall-clock runtime was **44.933 seconds**. The preserved standalone report was
`C:\Users\epicm\AppData\Local\Temp\hexgt_g7_epoch15_full.md`, 38,563
bytes, SHA-256
`6f644aae97705c923d89cae2718fa95d8a6f0f1a2b983beb9b7cb6df46aaa763`.

Two operational discrepancies were confirmed. First, the live
`epoch_*.pt` checkpoints were actually under
`runs/hexfield_eq_main_1/checkpoints`, rather than under the prefit path
implied by the task; the prefit arms use `checkpoint_epoch0.pt` and
`soak_init.pt` names. Second, the header comment in
`hexfield_eq_arm4_raylayout.env` still says ray-layout support is blocked,
but that comment is stale: its architecture values match the current code and
the strict-loaded live checkpoint exactly.

#### Provenance and contract

- Checkpoint: `E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\checkpoints\epoch_000015.pt`
- Checkpoint SHA-256: `c02c9f460d0adb12ecd0684dc48ab94ca4726f12e9c7a0139f19103f3a61728e`
- Checkpoint bytes: 8910678
- Checkpoint epoch: 15
- Architecture env: `E:\Hexo-BotTrainer-hexgt\scripts\prefit_env\hexfield_eq_arm4_raylayout.env`
- Position source: manifest/sidecar-uniform real rows from `E:\Hexo-BotTrainer\runs\hexfield_eq_main_1\samples`
- Positions audited: 512
- Fixed seed: 0
- Device: CPU only
- Triton modules imported: 0
- Production FlexAttention import: blocked; materialized CPU attention used
- Activation forward dtype: fp32 checkpoint inference
- Projection/energy accumulation dtype: fp64
- Architecture metadata strict match: PASS
- State-dict strict load: PASS

- Eligible committed shards: 4352
- Eligible manifest/sidecar rows: 399995
- Checkpoint train-state rows vs eligible source rows: 399995 vs 399995 (MATCH)

Node-count quartiles `(min, Q25, median, Q75, max)`: `(7.0, 287.0, 380.5, 534.5, 951.0)`.
Ply quartiles `(min, Q25, median, Q75, max)`: `(0.0, 24.0, 52.5, 103.25, 253.0)`.

##### Arm-4 architecture metadata

| Field | Value |
|---|---|
| `group_order` | `12` |
| `c_orbit` | `16` |
| `channels` | `192` |
| `in_channels` | `25` |
| `attention_heads` | `3` |
| `trunk_layout` | `CCLACCLACLA` |
| `num_tokens` | `6` |
| `feature_width` | `25` |
| `equivariant` | `True` |
| `reg_lane` | `True` |
| `reg_tok_read` | `False` |
| `support_radius` | `4` |
| `bias_reduction` | `joint_row_head` |
| `bias_joint_classes` | `81` |
| `ray_heads` | `6` |
| `ray_blockers` | `True` |

##### Parsed architecture env

| Variable | Value |
|---|---|
| `HEXFIELD_EQ_CHANNELS` | `192` |
| `HEXFIELD_EQ_GROUP_ORDER` | `12` |
| `HEXFIELD_EQ_C_ORBIT` | `16` |
| `HEXFIELD_EQ_ATTENTION_HEADS` | `3` |
| `HEXFIELD_EQ_SUPPORT_RADIUS` | `4` |
| `HEXFIELD_EQ_TRUNK` | `CCLACCLACLA` |
| `HEXFIELD_EQ_REG_LANE` | `1` |
| `HEXFIELD_EQ_REG_TOK_READ` | `0` |

#### Projection definition and internal checks

- `sigma = g7`; `rot180 = g3`.
- Right averaging: `(P_H v)[g] = mean_{h in H} v[g h]`.
- `G` blocks: `((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),)`
- `K` blocks: `((0, 3, 7, 10), (1, 4, 8, 11), (2, 5, 6, 9))`
- `mirror` blocks: `((0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 6))`
- `point` blocks: `((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))`
- Strengthened nesting asserted per stream and per orbit channel: `E_G <= E_K <= E_mirror` and `E_G <= E_K <= E_point`.
- Overall `E_H` is the energy-weighted ratio of sums. `macro E_H` is the unweighted mean over nonzero `(site, orbit_channel)` fiber vectors.

#### Cell-stream energy fractions

| Stream | Sites | Fiber vectors | E_G | E_K | E_mirror | E_point | macro E_mirror | Nesting |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `stem` | 213243 | 3411888 | 0.896526 | 0.923675 | 0.952424 | 0.943007 | 0.873202 | PASS |
| `depth_00_C` | 213243 | 3411888 | 0.902807 | 0.928529 | 0.955178 | 0.946945 | 0.913138 | PASS |
| `depth_01_C` | 213243 | 3411888 | 0.904188 | 0.929511 | 0.955885 | 0.947646 | 0.897673 | PASS |
| `depth_02_L` | 213243 | 3411888 | 0.904547 | 0.929965 | 0.954197 | 0.951866 | 0.894855 | PASS |
| `depth_03_A` | 213243 | 3411888 | 0.927912 | 0.948827 | 0.965662 | 0.966571 | 0.904826 | PASS |
| `depth_04_C` | 213243 | 3411888 | 0.932764 | 0.952865 | 0.968704 | 0.968181 | 0.908039 | PASS |
| `depth_05_C` | 213243 | 3411888 | 0.940464 | 0.958603 | 0.972547 | 0.972064 | 0.910727 | PASS |
| `depth_06_L` | 213243 | 3411888 | 0.930324 | 0.952398 | 0.967312 | 0.970094 | 0.902465 | PASS |
| `depth_07_A` | 213243 | 3411888 | 0.937035 | 0.956764 | 0.968859 | 0.975745 | 0.902190 | PASS |
| `depth_08_C` | 213243 | 3411888 | 0.953105 | 0.967577 | 0.978000 | 0.979216 | 0.907799 | PASS |
| `depth_09_L` | 213243 | 3411888 | 0.947939 | 0.965179 | 0.976191 | 0.978065 | 0.914736 | PASS |
| `depth_10_A` | 213243 | 3411888 | 0.951447 | 0.967291 | 0.977915 | 0.978786 | 0.923345 | PASS |
| `pre_ln_final` | 213243 | 3411888 | 0.951447 | 0.967291 | 0.977915 | 0.978786 | 0.923345 | PASS |
| `post_ln_final` | 213243 | 3411888 | 0.829422 | 0.887692 | 0.925544 | 0.926850 | 0.857930 | PASS |

#### Token-stream energy fractions

| Stream | Sites | Fiber vectors | E_G | E_K | E_mirror | E_point | macro E_mirror | Nesting |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `depth_00_C` | 3072 | 49152 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999833 | PASS |
| `depth_01_C` | 3072 | 49152 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999730 | PASS |
| `depth_02_L` | 3072 | 49152 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.999820 | PASS |
| `depth_03_A` | 3072 | 49152 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999656 | PASS |
| `depth_04_C` | 3072 | 49152 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999686 | PASS |
| `depth_05_C` | 3072 | 49152 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999735 | PASS |
| `depth_06_L` | 3072 | 49152 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 0.999615 | PASS |
| `depth_07_A` | 3072 | 49152 | 0.999990 | 0.999994 | 0.999994 | 1.000000 | 0.998930 | PASS |
| `depth_08_C` | 3072 | 49152 | 0.999990 | 0.999994 | 0.999994 | 1.000000 | 0.998765 | PASS |
| `depth_09_L` | 3072 | 49152 | 0.999992 | 0.999995 | 0.999995 | 1.000000 | 0.998926 | PASS |
| `depth_10_A` | 3072 | 49152 | 0.999992 | 0.999995 | 0.999995 | 1.000000 | 0.998294 | PASS |
| `pre_ln_final` | 3072 | 49152 | 0.999992 | 0.999995 | 0.999995 | 1.000000 | 0.998294 | PASS |
| `post_ln_final` | 3072 | 49152 | 0.999989 | 0.999994 | 0.999994 | 1.000000 | 0.998654 | PASS |

#### Per-channel distributions — cells

Each distribution contains the 16 orbit-channel energy fractions. The histogram intervals are left-closed and right-open except the final bin.

| Stream | H | Active | Min | Q25 | Median | Q75 | Max | >=0.70 | Histogram |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `stem` | G | 16/16 | 0.206117 | 0.581497 | 0.846872 | 0.910888 | 0.933662 | 9/16 | `[0.00,0.25):1 [0.25,0.50):3 [0.50,0.70):3 [0.70,0.80):0 [0.80,0.90):4 [0.90,0.95):5 [0.95,0.99):0 [0.99,1.00]:0` |
| `stem` | K | 16/16 | 0.479704 | 0.692183 | 0.889259 | 0.935073 | 0.958055 | 11/16 | `[0.00,0.25):0 [0.25,0.50):1 [0.50,0.70):4 [0.70,0.80):1 [0.80,0.90):3 [0.90,0.95):5 [0.95,0.99):2 [0.99,1.00]:0` |
| `stem` | mirror | 16/16 | 0.543314 | 0.858496 | 0.930355 | 0.958428 | 0.977304 | 14/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):2 [0.70,0.80):2 [0.80,0.90):2 [0.90,0.95):3 [0.95,0.99):7 [0.99,1.00]:0` |
| `stem` | point | 16/16 | 0.625766 | 0.886185 | 0.936530 | 0.963351 | 0.984925 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):3 [0.90,0.95):5 [0.95,0.99):6 [0.99,1.00]:0` |
| `depth_00_C` | G | 16/16 | 0.207454 | 0.717042 | 0.870370 | 0.907904 | 0.941207 | 13/16 | `[0.00,0.25):1 [0.25,0.50):1 [0.50,0.70):1 [0.70,0.80):3 [0.80,0.90):5 [0.90,0.95):5 [0.95,0.99):0 [0.99,1.00]:0` |
| `depth_00_C` | K | 16/16 | 0.535542 | 0.809797 | 0.900752 | 0.940906 | 0.964520 | 13/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):3 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):6 [0.95,0.99):2 [0.99,1.00]:0` |
| `depth_00_C` | mirror | 16/16 | 0.567606 | 0.904047 | 0.932703 | 0.963974 | 0.975908 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):6 [0.95,0.99):6 [0.99,1.00]:0` |
| `depth_00_C` | point | 16/16 | 0.626564 | 0.917870 | 0.954191 | 0.964183 | 0.984395 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):1 [0.90,0.95):4 [0.95,0.99):9 [0.99,1.00]:0` |
| `depth_01_C` | G | 16/16 | 0.200516 | 0.624588 | 0.846118 | 0.908870 | 0.942433 | 11/16 | `[0.00,0.25):1 [0.25,0.50):3 [0.50,0.70):1 [0.70,0.80):2 [0.80,0.90):4 [0.90,0.95):5 [0.95,0.99):0 [0.99,1.00]:0` |
| `depth_01_C` | K | 16/16 | 0.554088 | 0.703534 | 0.888166 | 0.940922 | 0.964428 | 13/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):3 [0.70,0.80):3 [0.80,0.90):3 [0.90,0.95):4 [0.95,0.99):3 [0.99,1.00]:0` |
| `depth_01_C` | mirror | 16/16 | 0.584966 | 0.858053 | 0.928118 | 0.964547 | 0.976071 | 14/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):2 [0.70,0.80):2 [0.80,0.90):1 [0.90,0.95):5 [0.95,0.99):6 [0.99,1.00]:0` |
| `depth_01_C` | point | 16/16 | 0.615768 | 0.902619 | 0.938227 | 0.962362 | 0.984603 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):3 [0.90,0.95):5 [0.95,0.99):7 [0.99,1.00]:0` |
| `depth_02_L` | G | 16/16 | 0.673676 | 0.788532 | 0.866454 | 0.913973 | 0.972328 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):3 [0.80,0.90):5 [0.90,0.95):5 [0.95,0.99):2 [0.99,1.00]:0` |
| `depth_02_L` | K | 16/16 | 0.702876 | 0.847867 | 0.909939 | 0.946656 | 0.990059 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):3 [0.80,0.90):5 [0.90,0.95):5 [0.95,0.99):2 [0.99,1.00]:1` |
| `depth_02_L` | mirror | 16/16 | 0.724882 | 0.894510 | 0.948351 | 0.967444 | 0.991975 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):3 [0.95,0.99):7 [0.99,1.00]:1` |
| `depth_02_L` | point | 16/16 | 0.798250 | 0.938145 | 0.965752 | 0.976965 | 0.997698 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):1 [0.90,0.95):3 [0.95,0.99):10 [0.99,1.00]:1` |
| `depth_03_A` | G | 16/16 | 0.654852 | 0.859394 | 0.902893 | 0.934566 | 0.991004 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):5 [0.90,0.95):5 [0.95,0.99):3 [0.99,1.00]:1` |
| `depth_03_A` | K | 16/16 | 0.688095 | 0.899777 | 0.939964 | 0.965522 | 0.997662 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):3 [0.90,0.95):7 [0.95,0.99):3 [0.99,1.00]:2` |
| `depth_03_A` | mirror | 16/16 | 0.705344 | 0.925991 | 0.964166 | 0.978648 | 0.998208 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):1 [0.90,0.95):4 [0.95,0.99):8 [0.99,1.00]:2` |
| `depth_03_A` | point | 16/16 | 0.895773 | 0.955204 | 0.969160 | 0.984585 | 0.999916 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):2 [0.95,0.99):9 [0.99,1.00]:3` |
| `depth_04_C` | G | 16/16 | 0.581968 | 0.822505 | 0.904390 | 0.920737 | 0.972822 | 14/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):2 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):7 [0.95,0.99):2 [0.99,1.00]:0` |
| `depth_04_C` | K | 16/16 | 0.661223 | 0.867270 | 0.934343 | 0.952903 | 0.997170 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):4 [0.95,0.99):5 [0.99,1.00]:1` |
| `depth_04_C` | mirror | 16/16 | 0.681359 | 0.918176 | 0.957614 | 0.969572 | 0.997210 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):3 [0.95,0.99):9 [0.99,1.00]:1` |
| `depth_04_C` | point | 16/16 | 0.811155 | 0.935808 | 0.966438 | 0.980944 | 0.999930 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):3 [0.90,0.95):2 [0.95,0.99):10 [0.99,1.00]:1` |
| `depth_05_C` | G | 16/16 | 0.574789 | 0.822088 | 0.906225 | 0.919526 | 0.978979 | 14/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):2 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):7 [0.95,0.99):2 [0.99,1.00]:0` |
| `depth_05_C` | K | 16/16 | 0.661664 | 0.866789 | 0.934327 | 0.953719 | 0.997207 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):5 [0.95,0.99):4 [0.99,1.00]:1` |
| `depth_05_C` | mirror | 16/16 | 0.681786 | 0.916836 | 0.958795 | 0.968453 | 0.997247 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):3 [0.95,0.99):9 [0.99,1.00]:1` |
| `depth_05_C` | point | 16/16 | 0.804388 | 0.935771 | 0.966734 | 0.983071 | 0.999931 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):3 [0.95,0.99):10 [0.99,1.00]:1` |
| `depth_06_L` | G | 16/16 | 0.655166 | 0.856764 | 0.893586 | 0.930849 | 0.974469 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):2 [0.80,0.90):5 [0.90,0.95):6 [0.95,0.99):2 [0.99,1.00]:0` |
| `depth_06_L` | K | 16/16 | 0.692217 | 0.896046 | 0.937948 | 0.954530 | 0.996331 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):3 [0.90,0.95):7 [0.95,0.99):4 [0.99,1.00]:1` |
| `depth_06_L` | mirror | 16/16 | 0.710870 | 0.922005 | 0.957097 | 0.971981 | 0.996379 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):4 [0.95,0.99):8 [0.99,1.00]:1` |
| `depth_06_L` | point | 16/16 | 0.906758 | 0.956563 | 0.969708 | 0.985344 | 0.999919 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):4 [0.95,0.99):9 [0.99,1.00]:3` |
| `depth_07_A` | G | 16/16 | 0.709871 | 0.882168 | 0.915686 | 0.953477 | 0.987145 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):2 [0.80,0.90):4 [0.90,0.95):5 [0.95,0.99):5 [0.99,1.00]:0` |
| `depth_07_A` | K | 16/16 | 0.729452 | 0.925136 | 0.955139 | 0.977748 | 0.992757 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):5 [0.95,0.99):7 [0.99,1.00]:1` |
| `depth_07_A` | mirror | 16/16 | 0.735662 | 0.957835 | 0.970603 | 0.981369 | 0.993277 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):1 [0.95,0.99):10 [0.99,1.00]:2` |
| `depth_07_A` | point | 16/16 | 0.925044 | 0.959032 | 0.982592 | 0.992121 | 0.999953 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):3 [0.95,0.99):7 [0.99,1.00]:6` |
| `depth_08_C` | G | 16/16 | 0.587849 | 0.873393 | 0.930671 | 0.961894 | 0.983760 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):3 [0.90,0.95):5 [0.95,0.99):6 [0.99,1.00]:0` |
| `depth_08_C` | K | 16/16 | 0.628526 | 0.919706 | 0.955684 | 0.980194 | 0.993298 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):5 [0.95,0.99):7 [0.99,1.00]:1` |
| `depth_08_C` | mirror | 16/16 | 0.643528 | 0.951464 | 0.973290 | 0.983638 | 0.993347 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):0 [0.80,0.90):2 [0.90,0.95):1 [0.95,0.99):9 [0.99,1.00]:3` |
| `depth_08_C` | point | 16/16 | 0.925523 | 0.956776 | 0.979784 | 0.992424 | 0.999971 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):3 [0.95,0.99):8 [0.99,1.00]:5` |
| `depth_09_L` | G | 16/16 | 0.681189 | 0.870047 | 0.929513 | 0.953625 | 0.984220 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):5 [0.95,0.99):5 [0.99,1.00]:0` |
| `depth_09_L` | K | 16/16 | 0.716303 | 0.917940 | 0.952431 | 0.974148 | 0.993453 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):5 [0.95,0.99):7 [0.99,1.00]:1` |
| `depth_09_L` | mirror | 16/16 | 0.728430 | 0.950574 | 0.969123 | 0.987682 | 0.993649 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):1 [0.95,0.99):10 [0.99,1.00]:2` |
| `depth_09_L` | point | 16/16 | 0.922355 | 0.956353 | 0.980377 | 0.992330 | 0.999969 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):2 [0.95,0.99):9 [0.99,1.00]:5` |
| `depth_10_A` | G | 16/16 | 0.681216 | 0.881940 | 0.923533 | 0.951033 | 0.990741 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):6 [0.95,0.99):3 [0.99,1.00]:1` |
| `depth_10_A` | K | 16/16 | 0.716284 | 0.938217 | 0.949612 | 0.973014 | 0.995369 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):5 [0.95,0.99):5 [0.99,1.00]:3` |
| `depth_10_A` | mirror | 16/16 | 0.728385 | 0.951683 | 0.970207 | 0.989997 | 0.995671 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):1 [0.90,0.95):2 [0.95,0.99):8 [0.99,1.00]:4` |
| `depth_10_A` | point | 16/16 | 0.941909 | 0.958975 | 0.979455 | 0.992407 | 0.999950 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):2 [0.95,0.99):9 [0.99,1.00]:5` |
| `pre_ln_final` | G | 16/16 | 0.681216 | 0.881940 | 0.923533 | 0.951033 | 0.990741 | 15/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):1 [0.70,0.80):1 [0.80,0.90):4 [0.90,0.95):6 [0.95,0.99):3 [0.99,1.00]:1` |
| `pre_ln_final` | K | 16/16 | 0.716284 | 0.938217 | 0.949612 | 0.973014 | 0.995369 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):5 [0.95,0.99):5 [0.99,1.00]:3` |
| `pre_ln_final` | mirror | 16/16 | 0.728385 | 0.951683 | 0.970207 | 0.989997 | 0.995671 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):1 [0.80,0.90):1 [0.90,0.95):2 [0.95,0.99):8 [0.99,1.00]:4` |
| `pre_ln_final` | point | 16/16 | 0.941909 | 0.958975 | 0.979455 | 0.992407 | 0.999950 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):2 [0.95,0.99):9 [0.99,1.00]:5` |
| `post_ln_final` | G | 16/16 | 0.422096 | 0.624690 | 0.809634 | 0.958555 | 0.997041 | 9/16 | `[0.00,0.25):0 [0.25,0.50):1 [0.50,0.70):6 [0.70,0.80):1 [0.80,0.90):2 [0.90,0.95):1 [0.95,0.99):4 [0.99,1.00]:1` |
| `post_ln_final` | K | 16/16 | 0.630604 | 0.762816 | 0.876385 | 0.969068 | 0.997990 | 13/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):3 [0.70,0.80):3 [0.80,0.90):3 [0.90,0.95):1 [0.95,0.99):4 [0.99,1.00]:2` |
| `post_ln_final` | mirror | 16/16 | 0.672410 | 0.856580 | 0.949713 | 0.972145 | 0.998138 | 14/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):2 [0.70,0.80):0 [0.80,0.90):5 [0.90,0.95):1 [0.95,0.99):6 [0.99,1.00]:2` |
| `post_ln_final` | point | 16/16 | 0.728687 | 0.840904 | 0.948723 | 0.997883 | 0.999791 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):3 [0.80,0.90):3 [0.90,0.95):2 [0.95,0.99):2 [0.99,1.00]:6` |

#### Per-channel distributions — tokens

Each distribution contains the 16 orbit-channel energy fractions. The histogram intervals are left-closed and right-open except the final bin.

| Stream | H | Active | Min | Q25 | Median | Q75 | Max | >=0.70 | Histogram |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `depth_00_C` | G | 16/16 | 0.999997 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_00_C` | K | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_00_C` | mirror | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_00_C` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_01_C` | G | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_01_C` | K | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_01_C` | mirror | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_01_C` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_02_L` | G | 16/16 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_02_L` | K | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_02_L` | mirror | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_02_L` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_03_A` | G | 16/16 | 0.999993 | 0.999999 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_03_A` | K | 16/16 | 0.999996 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_03_A` | mirror | 16/16 | 0.999996 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_03_A` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_04_C` | G | 16/16 | 0.999993 | 0.999999 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_04_C` | K | 16/16 | 0.999996 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_04_C` | mirror | 16/16 | 0.999996 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_04_C` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_05_C` | G | 16/16 | 0.999992 | 0.999999 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_05_C` | K | 16/16 | 0.999995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_05_C` | mirror | 16/16 | 0.999995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_05_C` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_06_L` | G | 16/16 | 0.999992 | 0.999998 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_06_L` | K | 16/16 | 0.999995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_06_L` | mirror | 16/16 | 0.999995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_06_L` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_07_A` | G | 16/16 | 0.999605 | 0.999981 | 0.999995 | 0.999998 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_07_A` | K | 16/16 | 0.999702 | 0.999993 | 0.999999 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_07_A` | mirror | 16/16 | 0.999702 | 0.999993 | 0.999999 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_07_A` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_08_C` | G | 16/16 | 0.999711 | 0.999983 | 0.999996 | 0.999998 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_08_C` | K | 16/16 | 0.999783 | 0.999995 | 0.999999 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_08_C` | mirror | 16/16 | 0.999783 | 0.999995 | 0.999999 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_08_C` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_09_L` | G | 16/16 | 0.999806 | 0.999989 | 0.999998 | 0.999998 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_09_L` | K | 16/16 | 0.999854 | 0.999997 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_09_L` | mirror | 16/16 | 0.999854 | 0.999997 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_09_L` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_10_A` | G | 16/16 | 0.999807 | 0.999988 | 0.999996 | 0.999998 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_10_A` | K | 16/16 | 0.999855 | 0.999996 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_10_A` | mirror | 16/16 | 0.999855 | 0.999996 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `depth_10_A` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `pre_ln_final` | G | 16/16 | 0.999807 | 0.999988 | 0.999996 | 0.999998 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `pre_ln_final` | K | 16/16 | 0.999855 | 0.999996 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `pre_ln_final` | mirror | 16/16 | 0.999855 | 0.999996 | 0.999999 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `pre_ln_final` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `post_ln_final` | G | 16/16 | 0.999849 | 0.999970 | 0.999991 | 0.999997 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `post_ln_final` | K | 16/16 | 0.999883 | 0.999989 | 0.999996 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `post_ln_final` | mirror | 16/16 | 0.999883 | 0.999989 | 0.999996 | 0.999999 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |
| `post_ln_final` | point | 16/16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16/16 | `[0.00,0.25):0 [0.25,0.50):0 [0.50,0.70):0 [0.70,0.80):0 [0.80,0.90):0 [0.90,0.95):0 [0.95,0.99):0 [0.99,1.00]:16` |

#### Optional EquivLinear weight-space audit

Each `wb[:, out_orbit, in_orbit]` is treated as a function on D6 and projected by the same right-H averages.

| Module | Vectors | E_G | E_K | E_mirror | E_point | mirror Q25 | mirror median | mirror Q75 | Nesting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `attn_blocks.0.attn.k_proj` | 256 | 0.479152 | 0.697527 | 0.748075 | 0.897940 | 0.492004 | 0.725938 | 0.892223 | PASS |
| `attn_blocks.0.attn.out_proj` | 256 | 0.666447 | 0.867657 | 0.911462 | 0.914700 | 0.632700 | 0.831636 | 0.933172 | PASS |
| `attn_blocks.0.attn.q_proj` | 256 | 0.508777 | 0.746738 | 0.802270 | 0.889401 | 0.571569 | 0.764983 | 0.886369 | PASS |
| `attn_blocks.0.attn.v_proj` | 256 | 0.679747 | 0.816912 | 0.847685 | 0.936754 | 0.617146 | 0.789745 | 0.918665 | PASS |
| `attn_blocks.0.fc1` | 512 | 0.658843 | 0.826732 | 0.842449 | 0.970365 | 0.708670 | 0.892579 | 0.960891 | PASS |
| `attn_blocks.0.fc2` | 512 | 0.613468 | 0.837125 | 0.852333 | 0.971649 | 0.666053 | 0.861545 | 0.951378 | PASS |
| `attn_blocks.1.attn.k_proj` | 256 | 0.499882 | 0.724629 | 0.783013 | 0.878314 | 0.584867 | 0.752472 | 0.871438 | PASS |
| `attn_blocks.1.attn.out_proj` | 256 | 0.686823 | 0.891078 | 0.925417 | 0.930659 | 0.723215 | 0.896870 | 0.949406 | PASS |
| `attn_blocks.1.attn.q_proj` | 256 | 0.473916 | 0.709160 | 0.760852 | 0.892230 | 0.533009 | 0.732808 | 0.857015 | PASS |
| `attn_blocks.1.attn.v_proj` | 256 | 0.750111 | 0.861167 | 0.885335 | 0.950243 | 0.618791 | 0.798224 | 0.911984 | PASS |
| `attn_blocks.1.fc1` | 512 | 0.637056 | 0.848327 | 0.868823 | 0.963241 | 0.714856 | 0.897559 | 0.967295 | PASS |
| `attn_blocks.1.fc2` | 512 | 0.676820 | 0.825542 | 0.842774 | 0.962279 | 0.628761 | 0.868365 | 0.957487 | PASS |
| `attn_blocks.2.attn.k_proj` | 256 | 0.778745 | 0.841310 | 0.857638 | 0.967209 | 0.633984 | 0.839057 | 0.944182 | PASS |
| `attn_blocks.2.attn.out_proj` | 256 | 0.719720 | 0.803865 | 0.869340 | 0.873964 | 0.584091 | 0.752037 | 0.903075 | PASS |
| `attn_blocks.2.attn.q_proj` | 256 | 0.847974 | 0.895370 | 0.918812 | 0.950179 | 0.666238 | 0.882986 | 0.951878 | PASS |
| `attn_blocks.2.attn.v_proj` | 256 | 0.805882 | 0.879983 | 0.904047 | 0.952684 | 0.659203 | 0.838442 | 0.930297 | PASS |
| `attn_blocks.2.fc1` | 512 | 0.540169 | 0.744268 | 0.789513 | 0.917919 | 0.658675 | 0.847191 | 0.946900 | PASS |
| `attn_blocks.2.fc2` | 512 | 0.622221 | 0.751544 | 0.815262 | 0.892237 | 0.623589 | 0.816787 | 0.928826 | PASS |
| `cell_q_expand` | 512 | 0.861516 | 0.888608 | 0.925139 | 0.926459 | 0.679674 | 0.855023 | 0.946459 | PASS |
| `inv_read` | 1024 | 0.740189 | 0.787001 | 0.859483 | 0.858289 | 0.599175 | 0.780688 | 0.901323 | PASS |
| `opp_policy_expand` | 512 | 0.768373 | 0.809658 | 0.875463 | 0.876945 | 0.467164 | 0.677282 | 0.856674 | PASS |
| `policy_expand` | 512 | 0.834533 | 0.861846 | 0.906561 | 0.907806 | 0.521957 | 0.770966 | 0.936123 | PASS |
| `ray_blocks.0.attn.k_proj` | 256 | 0.401704 | 0.674502 | 0.723869 | 0.917151 | 0.582564 | 0.780276 | 0.904055 | PASS |
| `ray_blocks.0.attn.out_proj` | 256 | 0.435135 | 0.898464 | 0.922017 | 0.953751 | 0.817387 | 0.900847 | 0.948503 | PASS |
| `ray_blocks.0.attn.q_proj` | 256 | 0.371743 | 0.684950 | 0.748134 | 0.877489 | 0.574312 | 0.770141 | 0.899461 | PASS |
| `ray_blocks.0.attn.v_proj` | 256 | 0.425172 | 0.704299 | 0.727408 | 0.954393 | 0.515948 | 0.743987 | 0.879253 | PASS |
| `ray_blocks.0.fc1` | 512 | 0.563797 | 0.811247 | 0.829691 | 0.965323 | 0.684543 | 0.880882 | 0.953272 | PASS |
| `ray_blocks.0.fc2` | 512 | 0.735273 | 0.831197 | 0.840630 | 0.981334 | 0.631149 | 0.861614 | 0.954962 | PASS |
| `ray_blocks.1.attn.k_proj` | 256 | 0.708296 | 0.818210 | 0.830094 | 0.976813 | 0.633824 | 0.837783 | 0.954084 | PASS |
| `ray_blocks.1.attn.out_proj` | 256 | 0.403953 | 0.876935 | 0.912715 | 0.927182 | 0.754242 | 0.886957 | 0.943452 | PASS |
| `ray_blocks.1.attn.q_proj` | 256 | 0.730111 | 0.870368 | 0.882072 | 0.979112 | 0.744261 | 0.917439 | 0.974604 | PASS |
| `ray_blocks.1.attn.v_proj` | 256 | 0.428247 | 0.740439 | 0.769896 | 0.939538 | 0.585366 | 0.762217 | 0.885767 | PASS |
| `ray_blocks.1.fc1` | 512 | 0.620989 | 0.820766 | 0.835565 | 0.970489 | 0.675403 | 0.879820 | 0.961941 | PASS |
| `ray_blocks.1.fc2` | 512 | 0.688628 | 0.867157 | 0.885047 | 0.967896 | 0.680618 | 0.887213 | 0.963509 | PASS |
| `ray_blocks.2.attn.k_proj` | 256 | 0.563550 | 0.767252 | 0.790873 | 0.951980 | 0.521502 | 0.782078 | 0.911231 | PASS |
| `ray_blocks.2.attn.out_proj` | 256 | 0.444854 | 0.872753 | 0.906943 | 0.934175 | 0.727992 | 0.878104 | 0.948887 | PASS |
| `ray_blocks.2.attn.q_proj` | 256 | 0.647522 | 0.811819 | 0.835925 | 0.951343 | 0.660360 | 0.824868 | 0.945763 | PASS |
| `ray_blocks.2.attn.v_proj` | 256 | 0.547486 | 0.799878 | 0.825650 | 0.948299 | 0.601873 | 0.825087 | 0.913642 | PASS |
| `ray_blocks.2.fc1` | 512 | 0.586648 | 0.751745 | 0.839276 | 0.863347 | 0.692287 | 0.859267 | 0.949732 | PASS |
| `ray_blocks.2.fc2` | 512 | 0.589213 | 0.669935 | 0.812931 | 0.810903 | 0.618661 | 0.817918 | 0.925608 | PASS |
| `registers.0.k_proj` | 256 | 0.876060 | 0.912797 | 0.941972 | 0.943841 | 0.725739 | 0.861405 | 0.953449 | PASS |
| `registers.0.out_proj` | 256 | 0.989783 | 0.991578 | 0.994402 | 0.994492 | 0.933400 | 0.985213 | 0.995386 | PASS |
| `registers.0.q_proj` | 256 | 0.867579 | 0.891687 | 0.929026 | 0.927850 | 0.693338 | 0.871510 | 0.950537 | PASS |
| `registers.0.v_proj` | 256 | 0.769930 | 0.814880 | 0.883818 | 0.873399 | 0.647748 | 0.784301 | 0.906900 | PASS |
| `registers.1.k_proj` | 256 | 0.843188 | 0.880703 | 0.914649 | 0.928541 | 0.645661 | 0.829320 | 0.936682 | PASS |
| `registers.1.out_proj` | 256 | 0.994670 | 0.995634 | 0.997193 | 0.997017 | 0.965915 | 0.993600 | 0.998290 | PASS |
| `registers.1.q_proj` | 256 | 0.864089 | 0.887604 | 0.924787 | 0.925399 | 0.670213 | 0.868035 | 0.955380 | PASS |
| `registers.1.v_proj` | 256 | 0.837809 | 0.866125 | 0.914484 | 0.911623 | 0.617562 | 0.826825 | 0.936061 | PASS |
| `registers.2.k_proj` | 256 | 0.906187 | 0.924528 | 0.945656 | 0.955044 | 0.675557 | 0.889367 | 0.960999 | PASS |
| `registers.2.out_proj` | 256 | 0.995191 | 0.996118 | 0.997587 | 0.997316 | 0.983292 | 0.994995 | 0.998556 | PASS |
| `registers.2.q_proj` | 256 | 0.890908 | 0.910244 | 0.941357 | 0.939129 | 0.676572 | 0.883513 | 0.952332 | PASS |
| `registers.2.v_proj` | 256 | 0.848223 | 0.876343 | 0.916064 | 0.918603 | 0.612768 | 0.827035 | 0.940514 | PASS |
| `registers.3.k_proj` | 256 | 0.918189 | 0.933055 | 0.954415 | 0.954654 | 0.827819 | 0.940795 | 0.974130 | PASS |
| `registers.3.out_proj` | 256 | 0.988727 | 0.990936 | 0.993775 | 0.994022 | 0.903155 | 0.983529 | 0.994854 | PASS |
| `registers.3.q_proj` | 256 | 0.932207 | 0.944072 | 0.962626 | 0.962940 | 0.748770 | 0.930134 | 0.975606 | PASS |
| `registers.3.v_proj` | 256 | 0.793892 | 0.830099 | 0.886456 | 0.885991 | 0.589989 | 0.775550 | 0.900028 | PASS |
| `registers.4.k_proj` | 256 | 0.830226 | 0.861243 | 0.911092 | 0.905807 | 0.730830 | 0.875067 | 0.938337 | PASS |
| `registers.4.out_proj` | 256 | 0.985146 | 0.987834 | 0.991916 | 0.991752 | 0.914430 | 0.977411 | 0.995059 | PASS |
| `registers.4.q_proj` | 256 | 0.926756 | 0.939477 | 0.959914 | 0.961126 | 0.769031 | 0.919077 | 0.973110 | PASS |
| `registers.4.v_proj` | 256 | 0.735596 | 0.783995 | 0.856851 | 0.855756 | 0.592666 | 0.744965 | 0.897801 | PASS |
| `registers_l.0.k_proj` | 256 | 0.714916 | 0.761261 | 0.845989 | 0.839503 | 0.600568 | 0.794294 | 0.902089 | PASS |
| `registers_l.0.out_proj` | 256 | 0.989541 | 0.991542 | 0.994227 | 0.994374 | 0.946726 | 0.984859 | 0.995195 | PASS |
| `registers_l.0.q_proj` | 256 | 0.772313 | 0.811702 | 0.874650 | 0.876083 | 0.564055 | 0.739177 | 0.878768 | PASS |
| `registers_l.0.v_proj` | 256 | 0.756611 | 0.802851 | 0.867325 | 0.867894 | 0.556363 | 0.759266 | 0.892204 | PASS |
| `registers_l.1.k_proj` | 256 | 0.833031 | 0.863768 | 0.909085 | 0.909768 | 0.641703 | 0.834217 | 0.941680 | PASS |
| `registers_l.1.out_proj` | 256 | 0.978399 | 0.982067 | 0.988039 | 0.988118 | 0.890463 | 0.971254 | 0.991378 | PASS |
| `registers_l.1.q_proj` | 256 | 0.817928 | 0.851042 | 0.902362 | 0.899256 | 0.655437 | 0.842403 | 0.922175 | PASS |
| `registers_l.1.v_proj` | 256 | 0.628260 | 0.694875 | 0.788919 | 0.799332 | 0.488051 | 0.661448 | 0.807696 | PASS |
| `registers_l.2.k_proj` | 256 | 0.901367 | 0.920563 | 0.946591 | 0.947363 | 0.722376 | 0.904183 | 0.966958 | PASS |
| `registers_l.2.out_proj` | 256 | 0.987785 | 0.989918 | 0.993188 | 0.993201 | 0.911653 | 0.980833 | 0.995430 | PASS |
| `registers_l.2.q_proj` | 256 | 0.896361 | 0.914059 | 0.941903 | 0.943881 | 0.715845 | 0.900889 | 0.958107 | PASS |
| `registers_l.2.v_proj` | 256 | 0.719119 | 0.772256 | 0.846294 | 0.845273 | 0.547598 | 0.766575 | 0.897097 | PASS |
| `soft_policy_expand` | 512 | 0.694653 | 0.753423 | 0.836519 | 0.833946 | 0.456024 | 0.639424 | 0.815096 | PASS |

#### G7 interpretation

- Mirror-invariant energy is at least 70% in 11/11 trunk-depth cell streams.
- Mean mirror-invariant energy across trunk depth: 0.967314.
- Owner rule (`E_mirror >= 70%` across most trunk depth): **STRONG GO SIGNAL**.
- This is the G7 representation-evidence signal only; the Phase-A GO/NO-GO decision must also use G8 costs and all earlier gates.

#### Acceptance checks

- Every expected cell stream observed (14): PASS
- Every expected token stream observed (13): PASS
- All subgroup energy bounds and strengthened nesting relations: PASS
- CPU/no-Triton runtime contract: PASS
- Deterministic seed and manifest-uniform sampling: PASS

Later gate results, audit evidence, cost rankings, and the final recommendation
will be filled as each ordered gate turns green.
