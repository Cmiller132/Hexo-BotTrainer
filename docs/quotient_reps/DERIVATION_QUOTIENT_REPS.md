# Derivation — quotient-permutation fibers for exact D6 equivariance

Status: **Phase-A CPU proof / Phase-B boundary contract.** Date: 2026-07-09.
This document extends `docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` without
changing its conventions. Phase A proves the representation machinery and the
boundary design; it does **not** implement Phase B in the production model.

The executable proof is `packages/hexfield_eq/python/hexfield_eq/reps.py`. Its
claims are checked in fp64 by:

| gate | machine check |
|---|---|
| G1, group and quotient actions | `tests/test_hexfield_eq_reps_group.py` |
| G2, linear and conv dimensions | `tests/test_hexfield_eq_reps_homdims.py` |
| G3, production specialization | `tests/test_hexfield_eq_reps_parity.py` |
| G4–G5, typed layers and nonlinearities | `tests/test_hexfield_eq_reps_typed_layers.py` |
| G6, end-to-end boundary rehearsal | `tests/test_hexfield_eq_reps_toynet.py` |

All group tables in `reps.py` are regenerated from `geometry.apply_d6`; none is
copied from `equivariant.py`. Exact comparison to `equivariant.build_group()` is
a test, so the production convention is an independently checked result rather
than an assumption (`test_generated_group_tables_exactly_match_production`).

---

## 0. Conventions inherited from the production derivation

Let `G = D6`, the order-12 symmetry group of the hex lattice. Element indices
`0..11`, coordinate actions, and composition are exactly those of
`geometry.apply_d6`. In particular, `0..5` are the six rotations and `6..11`
are the six reflections. The generated multiplication table uses

```
mult[a][b] = a ∘ b,
```

so `a` acts after `b`. Tap `0` is the center and taps `1..6` are
`constants.DIRECTIONS`; write `τ_g(t)` for their generated permutation.

For a subgroup `H ≤ G`, let

```
X_H = G/H = {xH : x ∈ G}
```

be the set of **left cosets**. The quotient representation `V_H = R[X_H]`
acts by left translation. If `π_H(g)` maps source slot to destination slot,

```
π_H(g)(xH) = gxH,
ρ_H(g)e_s = e_{π_H(g)s},
(ρ_H(g)v)_s = v_{π_H(g^{-1})s}.                 (0.1)
```

The last line is the gather convention. Thus `rep_action` is forward
source-to-destination and `rep_gather(type,g) = rep_action(type,g^{-1})`.
Every `ρ_H(g)` is an orthogonal 0/1 permutation matrix and
`ρ_H(g)ρ_H(h)=ρ_H(gh)`.

For a spatial feature field `f`, the action remains the one in the existing
derivation:

```
(T_g f)(x) = ρ(g) f(g^{-1}x).                    (0.2)
```

Only the fiber representation `ρ` is generalized below.

---

## 1. Quotient types and the canonical mixed layout

### 1.1 The distinguished subgroups and the one-vector correction

The Q-axis stabilizer is derived geometrically, not imported:

```
K_axis = {g : g·(1,0) ∈ {(1,0),(-1,0)}} = {0,3,7,10}.
```

The determinant of the axial-basis action distinguishes rotations (`+1`) from
reflections (`−1`). Inside `K_axis`:

| element | `(1,0)` image | `(0,1)` image | orientation |
|---|---:|---:|---:|
| `g0` | `(1,0)` | `(0,1)` | `+1` |
| `g3` | `(-1,0)` | `(0,-1)` | `+1` |
| `g7` | `(1,0)` | `(1,-1)` | `−1` |
| `g10` | `(-1,0)` | `(-1,1)` | `−1` |

Therefore the reflection in `K_axis` that fixes the **directed** Q vector is

```
σ = g7,
```

and the 180-degree rotation is

```
rot180 = g3.
```

There is a real ambiguity in the literal one-vector criterion in Phase-A §2:
both `g3` and `g10` map `(1,0)` to `(-1,0)`. The unambiguous definition adds
that `rot180` is orientation-preserving and reverses both axial basis vectors.
This selects `g3`; `g10` is a reflection. The correction is frozen by
`test_distinguished_subgroups_are_derived_from_geometry`.

### 1.2 The five types

Each required type is one transitive quotient-permutation module:

| type | defining subgroup `H` | slots `[G:H]` | intended meaning |
|---|---|---:|---|
| `reg` | `{0}` | 12 | fully chiral/orientation-sensitive |
| `mirror` | `⟨σ⟩ = {0,7}` | 6 | six poses modulo the base reflection; achiral carrier |
| `point` | `⟨rot180⟩ = {0,3}` | 6 | six poses modulo the 180-degree rotation |
| `axis` | `K_axis = {0,3,7,10}` | 3 | one slot per unoriented win axis |
| `triv` | `G` | 1 | fully D6-invariant scalar |

Within each coset the element indices are sorted; cosets are then ordered by
their minimum element. The nontrivial canonical slot lists are

```
mirror: ({0,7}, {1,8}, {2,9}, {3,10}, {4,11}, {5,6})
point : ({0,3}, {1,4}, {2,5}, {6,9}, {7,10}, {8,11})
axis  : ({0,3,7,10}, {1,4,8,11}, {2,5,6,9}).
```

The `axis` list is exactly production's Q/R/QR head-coset partition. Regular
slots are the twelve singleton cosets in index order; `triv` has the single
coset `G`. `test_canonical_coset_order` checks this construction, and
`test_all_quotient_actions_are_permutation_homomorphisms` checks all five types
and all `12²` group products.

### 1.3 Signatures and the resolved flattening discrepancy

A signature is a multiplicity map written in the fixed type order

```
reg, mirror, point, axis, triv.
```

Zero multiplicities are omitted. If type `T` has `S_T` slots and multiplicity
`m_T`, then

```
C(sig) = Σ_T S_T m_T,              n_instances(sig) = Σ_T m_T.   (1.1)
```

Type blocks are contiguous and canonical. **Inside a type block, the frozen
production-compatible layout is slot-major, instance-minor:**

```
channel(T, slot=s, instance=i) = offset_T + s·m_T + i.           (1.2)
```

Phase-A §2 contains an internal discrepancy: one sentence says
“instance-major, slot-minor” and gives `offset + instance·slots + slot`, while
constraint C4, `CONTEXT.md`, and the production regular fiber require
slot-major layout. Raw elementwise G3 parity and the Phase-B D8 checkpoint gate
are possible only with (1.2). The code-is-ground-truth rule therefore resolves
the sentence in favor of (1.2), for **every** type block. This is not a change
of representation up to basis permutation, but it is load-bearing for dense
weight order and checkpoint compatibility. The correction is frozen by
`test_regular_signature_layout_is_production_slot_major`.

The mixed action is block diagonal and never permutes the instance index:

```
(T,s,i)  ─ρ(g)→  (T, π_T(g)s, i).                              (1.3)
```

In gather form, channel `(T,s,i)` reads `(T,π_T(g^{-1})s,i)`. A vector with one
scalar per instance expands as `v[T,s,i]=v_base[T,i]`; it is fixed by all of G.

---

## 2. Equivariant linear maps: orbit basis = double cosets = Reynolds rank

Consider one input type `I=G/H_in` and one output type `O=G/H_out`, initially
with one instance of each. A linear map has matrix entries `W[a,b]`, where
`a∈O` and `b∈I`.

### 2.1 Orbit-basis theorem

`W` is equivariant exactly when

```
ρ_out(g) W = W ρ_in(g)                                  (2.1)
```

for every `g`. Because both actions are permutations, (2.1) is equivalent to

```
W[π_out(g)a, π_in(g)b] = W[a,b].                         (2.2)
```

Hence `W` is constant on the orbits of the diagonal action

```
g·(a,b) = (π_out(g)a, π_in(g)b)                          (2.3)
```

on `O×I`. Conversely, assigning one arbitrary scalar to each orbit satisfies
(2.2). The indicator matrices of the orbits have disjoint support, so they are
linearly independent. They form a basis of `Hom_G(V_in,V_out)`, and

```
dim Hom_G(V_in,V_out) = number of G-orbits on O×I.         (2.4)
```

For multiplicities `m_in,m_out`, every orbit coefficient becomes an arbitrary
`m_out×m_in` matrix. If `L[a,b]` is the canonical orbit label, the actual
slot-major materialization is

```
W[offset_out + a·m_out + o,
  offset_in  + b·m_in  + i] = θ[L[a,b], o, i].             (2.5)
```

For mixed signatures, sum (2.5) over every ordered type pair. Thus the total
number of scalar weights is

```
Σ_(I,O) d_linear[I,O] · m_I · m_O,                        (2.6)
```

plus one invariant bias per output instance.

### 2.2 Double-coset proof

Write a pair as `(xH_out,yH_in)`. The map

```
(xH_out,yH_in) ↦ H_out x⁻¹y H_in                         (2.7)
```

is well-defined: changing either coset representative only multiplies `x⁻¹y`
on the left by `H_out` or on the right by `H_in`. It is unchanged by the
diagonal left action because `(gx)⁻¹(gy)=x⁻¹y`. Every double coset is reached by
the pair `(H_out,zH_in)`, and equality of double cosets is exactly equality of
diagonal orbits. Therefore

```
G\(G/H_out × G/H_in)  ≅  H_out\G/H_in,                    (2.8)
dim Hom_G(V_in,V_out) = |H_out\G/H_in|.                   (2.9)
```

This is a second construction: `double_cosets` enumerates the sets directly,
independently of the orbit-label generator.

### 2.3 Reynolds-projector proof

On matrix space define

```
R_g(W) = ρ_out(g) W ρ_in(g)⁻¹.
```

Its fixed space is precisely (2.1). The group average

```
P(W) = (1/12) Σ_g R_g(W)                                 (2.10)
```

is the orthogonal projector onto that fixed space. With row-major
vectorization of `(out_slot,in_slot)`, permutation orthogonality gives

```
P = (1/12) Σ_g ρ_out(g) ⊗ ρ_in(g).                        (2.11)
```

Consequently `rank(P)=dim Hom_G(V_in,V_out)`. The fp64 test constructs this
dense projector, checks symmetry and idempotence to `1e-12`, and takes its SVD
rank with threshold `1e-9`. This calculation shares neither the union-find
orbit labels nor the direct double-coset enumeration.

### 2.4 Exact dimension table

Rows are input types and columns are output types, both in canonical order.
Every entry below agrees by all three methods (2.4), (2.9), and (2.11):

| input ↓ / output → | `reg` | `mirror` | `point` | `axis` | `triv` |
|---|---:|---:|---:|---:|---:|
| `reg` | 12 | 6 | 6 | 3 | 1 |
| `mirror` | 6 | 4 | 3 | 2 | 1 |
| `point` | 6 | 3 | 6 | 3 | 1 |
| `axis` | 3 | 2 | 3 | 2 | 1 |
| `triv` | 1 | 1 | 1 | 1 | 1 |

The anchors follow immediately: `reg→reg` has 12 basis elements;
`reg↔T` has `slots(T)`; and `triv→triv` has one. The complete three-way check
is `test_linear_dimensions_agree_three_independent_ways`.

### 2.5 Production regular specialization

For regular input and output, (2.2) says that the matrix depends only on the
relative group element:

```
W[out=a,in=b] = wb[a⁻¹b].                                  (2.12)
```

The canonical orbit labels have a transversal `(out=e,in=s)`. Mapping the
twelve production `wb[s]` blocks through this transversal is a bijection to
the generated twelve basis coefficients. After the slot-major expansion,
`typed_linear_weight` is elementwise identical to
`equivariant.gen_linear_weight`, not merely equivalent up to a permutation.
`test_typed_linear_exactly_reproduces_production` checks random fp64 weights
with `atol=rtol=0`.

---

## 3. Seven-tap typed convolution

Let `δ_t` be the center-plus-six-neighbor tap offsets, and write the mathematical
kernel matrix at tap `t` as `W_t : V_in→V_out`. The convolution is

```
(Cf)(x) = Σ_t W_t f(x+δ_t) + b.
```

Substitute the field action (0.2) and reindex offsets by
`gδ_t=δ_{τ_g(t)}`. Equivariance is equivalent to

```
W_{τ_g(t)} = ρ_out(g) W_t ρ_in(g)⁻¹,          for every g,t,  (3.1)
b            = ρ_out(g)b.                                  (3.2)
```

Equation (3.2) gives one bias scalar per output instance, copied across all
slots of that instance.

### 3.1 Tap-orbit basis and Reynolds projector

In entries, (3.1) says coefficients are constant on diagonal orbits of

```
g·(t,a,b) = (τ_g(t), π_out(g)a, π_in(g)b)                   (3.3)
```

in `Taps×O×I`. The orbit-indicator tensors are therefore a basis of all typed
equivariant seven-tap kernels. If `L_conv[t,a,b]` is the canonical label, each
label again carries an arbitrary `m_out×m_in` coefficient block. The generator
materializes the production layout `(7,C_in,C_out)`; the matrices in (3.1) use
the conventional transposed `(C_out,C_in)` view at each tap.

The independent averaging proof uses

```
(P_conv W)_t = (1/12) Σ_g
    ρ_out(g) W⁰_{τ_{g⁻¹}(t)} ρ_in(g)⁻¹.                    (3.4)
```

Equivalently, on row-major entry space this is the average of the permutation
action `τ(g)⊗ρ_out(g)⊗ρ_in(g)`. Its image is exactly (3.1), so its rank equals
the triple-orbit count. `test_conv_dimensions_agree_between_orbits_and_reynolds_rank`
checks both methods for all 25 ordered type pairs.

### 3.2 Exact conv dimension table

Rows are input types and columns are output types:

| input ↓ / output → | `reg` | `mirror` | `point` | `axis` | `triv` |
|---|---:|---:|---:|---:|---:|
| `reg` | 84 | 42 | 42 | 21 | 7 |
| `mirror` | 42 | 24 | 21 | 12 | 5 |
| `point` | 42 | 21 | 24 | 12 | 4 |
| `axis` | 21 | 12 | 12 | 7 | 3 |
| `triv` | 7 | 5 | 4 | 3 | 2 |

For `reg→reg`, the diagonal action on the two regular slots is free: a group
element fixing either regular slot is the identity. Every triple orbit thus has
size 12, giving the required anchor

```
dim Conv_G(reg,reg) = 7·12·12 / 12 = 84.                   (3.5)
```

This is also `12` center-tap blocks plus `72` direction-tap blocks, exactly the
existing production parameterization `w_base (7,12,m_out,m_in)`. A transversal
is `(tap=t,out=e,in=s)`, and the generated weight reduces to

```
W_t[out=a,in=b] = w_base[τ_{a⁻¹}(t), a⁻¹b].                (3.6)
```

`production_conv_coefficients` implements the explicit bijection between the
84 production blocks and the canonical triple-orbit labels.
`test_typed_conv_exactly_reproduces_production` checks the two materialized
weights elementwise with `atol=rtol=0`.

---

## 4. Pointwise nonlinearities: the precise legality statement

For a scalar function `φ:R→R`, let its shared componentwise lift be

```
Φ(x)_j = φ(x_j).
```

**Permutation theorem.** For every permutation matrix `P` and every scalar
function `φ`, without assumptions such as linearity, oddness, or smoothness,

```
Φ(Px) = PΦ(x).                                             (4.1)
```

This is simply relabeling coordinates before or after applying the same scalar
rule. All five quotient types and every mixed signature are permutation
representations, so shared pointwise ReLU and GELU are equivariant.

There is a useful converse, but its quantifier matters. Suppose an invertible
linear action `A` must commute with **every** shared componentwise scalar
function. The constant function `φ≡1` forces `A1=1`. The function `φ(t)=t²`
then gives, row by row and for arbitrary `x`,

```
(Σ_j a_j x_j)² = Σ_j a_j x_j².
```

Comparing coefficients forces `a_j²=a_j` and `a_ja_k=0` for `j≠k`; the row-sum
condition leaves exactly one `1` per row. Invertibility leaves exactly one per
column, so `A` is a permutation. Thus permutation actions are exactly the
actions that support **unrestricted** shared componentwise nonlinearities.

This does **not** say that permutations are the only representations compatible
with each particular nonlinearity. For example, signed permutations commute
with an odd scalar function, and irrep networks can use norms, gates, tensor
products, or other specially designed nonlinearities. The claim needed here is
that the existing unrestricted per-channel GELU/ReLU remains legal without such
machinery.

### 4.1 Sign-representation negative control

The one-dimensional orientation character is a valid representation:

```
χ(g) = det(g) = +1 for rotations, −1 for reflections,
χ(gh) = χ(g)χ(h).
```

GELU is not odd, so for a reflection

```
GELU(χ(g)x) = GELU(−x) ≠ −GELU(x) = χ(g)GELU(x).           (4.2)
```

`test_sign_rep_is_a_representation_but_gelu_breaks_it` checks all 144
representation products and the vector `[-2,-0.75,0.25,1.5]`; each of the six
reflections has maximum violation greater than `1.0`, while all six rotations
have exactly zero violation. The positive control checks all five quotient
types and all twelve group elements. This is why the sign rep is excluded from
the current type system even though it is mathematically a valid D6 rep.

---

## 5. Typed normalization, LayerScale, bias, and pooling

Let a mixed signature have width `C`, and let `ρ(g)` be its channel
permutation.

### 5.1 Full-fiber affine LayerNorm

Full-fiber mean and variance are symmetric functions of the `C` channels:

```
μ(ρx)=μ(x),                 var(ρx)=var(x).
```

Therefore the normalized vector transforms by `ρ`. An affine remains
equivariant when its parameters are fixed by `ρ`. Because each quotient action
is transitive on the slots of one instance, the fixed vectors are exactly the
slot-constant ones:

```
γ[T,s,i] = γ_base[T,i],       β[T,s,i] = β_base[T,i].       (5.1)
```

In layout (1.2), `instance_of_channel` expands one `n_instances(sig)` vector
over the slots of each type block. Hence

```
LN_typed(ρx) = ρ LN_typed(x).                              (5.2)
```

The implementation deliberately normalizes over the whole mixed fiber, not
separately by type, matching production LayerNorm and preserving the existing
fused-kernel interface. `TypedGroupAffineNorm.weight` and `.bias` are expanded
`(C,)` views.

### 5.2 LayerScale and linear bias

The same fixed-vector argument applies to residual LayerScale and to a linear
or conv bias:

```
LS(x)[T,s,i] = γ_base[T,i] x[T,s,i],
b[T,s,i]     = b_base[T,i].                                (5.3)
```

There is one learned scalar per type instance, never one independent scalar per
slot.

### 5.3 Typed group pool

For each type instance, define

```
pool(x)[T,i] = (1/S_T) Σ_(s=0)^(S_T−1) x[T,s,i].            (5.4)
```

Every group element only permutes this sum, so

```
pool(ρ(g)x) = pool(x).                                     (5.5)
```

The output is a trivial vector of width `n_instances(sig)`. This is the only
legal boundary to an unconstrained `nn.Linear` readout. It also explains why an
invariant learned seed has exactly one parameter per type instance: expansion
by (5.1) parameterizes the complete fixed subspace.

`test_fifty_random_signatures_norm_scale_pool_and_pointwise` checks typed norm,
LayerScale, pool, GELU, and ReLU for 50 seeded mixed signatures and all twelve
group elements. Linear and conv equivariance are checked over their own 50
seeded signature pairs in the same file.

---

## 6. Reynolds lift of the real 25-plane stem

The input is not a regular fiber. Its exact production plane action is

```
V_input = 13·triv ⊕ 4·axis.                                (6.1)
```

The physical layout is noncontiguous by type:

- scalar planes `0..10` and `23..24` are fixed;
- planes `11..22` are four quantity-major axis triples;
- `plane(q,a) = 11 + 3q + a`, with `q=0..3`, `a∈{Q,R,QR}`;
- `ρ_input(g)` sends `plane(q,a)` to
  `plane(q,π_axis(g)a)`.

The axis permutation is the same three-coset action used by the attention
heads. In particular, the input action must not be built as a fictitious
contiguous `13 scalars + 12 axes` block; the two fork scalars follow the axis
planes in the shipped layout.

Let `sig_out` be any mixed output signature and let an unrestricted seed have
mathematical matrices `W⁰_t` of shape `(C_out,25)`. Its orthogonal Reynolds
projection is

```
W̄_t = (1/12) Σ_g
    ρ_out(g) W⁰_{τ_{g⁻¹}(t)} ρ_input(g)⁻¹.                 (6.2)
```

Changing variables in the sum shows

```
W̄_{τ_h(t)} = ρ_out(h) W̄_t ρ_input(h)⁻¹,                  (6.3)
```

which is exactly the typed stem version of the conv constraint (3.1).
Conversely, every tensor satisfying (6.3) is fixed by (6.2), so the projection
is onto the complete equivariant stem space, not a hand-selected injection.
The seed parameter is stored as `w0 (7,C_out,25)` and the materialized dense
conv weight as `(7,25,C_out)`.

For pure regular output, `ρ_out` and `ρ_input` exactly match production's
generated matrices. `test_input_rep_and_typed_stem_reproduce_production` first
checks all twelve 25-plane matrices elementwise and then compares the projected
weights to `equivariant.gen_stem_weight` at `atol=1e-12, rtol=0`. The averaging
projection needs a rounding tolerance; the linear and conv orbit gathers in
§§2–3 do not.

---

## 7. Phase-B boundary design rehearsed by the typed toy network

This section records the locked choices D1–D8 from the Phase-B specification.
It is a boundary contract, not authorization to start Phase B. The two genuine
architecture choices left to the Phase-A evidence are the residual signature
`sig` and the regular attention multiplicity `K_attn`; the remaining mechanics
below are fixed.

G6 instantiates the contract at `K_attn=4`, hence internal regular width 48 and
A-head dimension 16, with two signatures:

```
reg:2,mirror:2,point:1,axis:2,triv:3   → C=51
reg:4,mirror:4,axis:4,triv:12          → C=96.
```

For each, the toy contains the real typed 25-plane stem, two two-conv residual
blocks, a sigmoid-gated register sum, one attention/MLP block, typed norms and
LayerScale, and policy/value reads. Five legal oracle-featurized positions are
checked under every group element: per-cell policy logits follow the transformed
node permutation and value is invariant at `atol=1e-9`
(`test_two_typed_toynets_are_equivariant_on_real_oracle_features`).

### 7.1 D1 — typed residual stream, regular attention internals

Only the residual stream carries `sig`. Define

```
R_attn = reg:K_attn,                  W_attn = 12·K_attn.
```

Every attention-like module has the boundary

```
sig ──TypedLinear(q,k,v)──> R_attn
    ──existing regular attention math──> R_attn
    ──TypedLinear(out)──> sig.                               (7.1)
```

Since both boundary maps are intertwiners, the internal tensor transforms by
the same regular action already proved in the production derivation. Residual
addition occurs only after returning to `sig`.

- A attention keeps three axis-coset heads and
  `head_dim_A=4·K_attn`. `head_perm(K_attn)` groups the regular slots by the
  three left cosets of `K_axis`. The joint `(pair-row,head)` bias LUT is
  signature-independent and unchanged.
- L/ray attention keeps six structural heads and
  `head_dim_L=2·K_attn`. The own/opp split rides the regular multiplicity index,
  as in production, so `K_attn` must be even. Ray bias and blocker semantics do
  not change.
- RegisterRefresh uses the same typed→regular q/k/v and regular→typed out
  boundaries around its sigmoid-gated **unnormalized sum**.

The G6 toy directly exercises the A and register boundaries. Ray attention uses
the same representation boundary but its six-head production integration
remains a Phase-B test obligation.

### 7.2 D2 — typed MLPs

For `MLP_RATIO=r`, multiply every type multiplicity, not every slot count by a
new rule:

```
fc1 : sig → r·sig,
GELU: componentwise on r·sig,
fc2 : r·sig → sig.                                         (7.2)
```

Here `r·sig` means `m_T↦r m_T` for each type. Section 4 makes GELU legal. The
toy rehearses (7.2) with `r=2`.

### 7.3 D3 — token typing

The learned token **seed** has shape

```
(NUM_TOKENS, n_instances(sig))
```

and expands slot-constant within each instance using (5.1). This is the complete
G-fixed subspace and generalizes the current `(NUM_TOKENS,C_ORBIT)` regular seed.
After tokens interact with a board, their dense activations occupy the full
typed fiber and transform by `ρ_sig`; they need not remain slot-constant for a
particular board. Token positions themselves do not move. The fp32 token-stream
carry remains unchanged. G6 constructs the seeds this way before register
refresh and joint token/cell attention.

### 7.4 D4 — per-instance norm and LayerScale

All norms use full mixed-fiber statistics with `gamma/beta` stored per instance
and expanded by `instance_of_channel`; LayerScale does the same. Expanded
`.weight` and `.bias` `(C,)` views must remain available to the fused conv+LN
kernel. Section 5 proves this and the G6 residual blocks exercise it.

### 7.5 D5 — typed convs and the typed stem

Residual convs use the triple-orbit basis of §3. The stem uses the 25-plane
Reynolds lift of §6. Materialized dense layouts remain `(7,C_in,C_out)` for
convs and `(C_out,C_in)` for linears, so serve caches, GEMMs, and Triton kernels
remain blind to the parameter tie. The pure-regular specialization must retain
the production `w_base` convention in (3.6).

### 7.6 D6 — invariant head boundaries

No unconstrained linear layer may consume a covariant typed fiber directly.

- A per-cell policy-style read is typed conv → typed expansion
  (`POLICY_READ_EXPAND·sig`) → `typed_group_pool` → plain `nn.Linear`. Its
  channel value is invariant while its cell index transforms, so policy logits
  permute with cells.
- Scalar reads first map to `INV_READ_EXPAND·sig`, pool every type instance,
  and only then use ordinary reductions/heads. Pooled cells and token
  activations are treated this way. Their plain-linear input width is
  `INV_READ_EXPAND·n_instances(sig)` per read block.

The G6 policy path uses expansion factor 2; its value path pools separate token
and cell reads before the final plain linear.

### 7.7 D7 — serve permutation folding stays on the regular side

The existing head reorder is folded only where a boundary tensor is regular:
q/k/v fold an output permutation and out-projection folds the corresponding
input permutation. The fold remains a row/column reorder of the materialized
dense boundary weight and retains the exact `not torch.is_grad_enabled()` cache
gate contract. G3 proves the generated dense pure-regular maps have production
ordering; the no-grad cache/fold integration itself remains a Phase-B serve
gate.

### 7.8 D8 — pure-regular bit-compatibility is the regression gate

With `sig=reg:16` and `K_attn=16`, Phase B must preserve the current state-dict
key set, parameter names, and shapes, including `wb (12,o,i)` and
`w_base (7,12,o,i)`. It must load the live checkpoint and reproduce
`forward_policy_value` on fp32 CPU to elementwise `atol≤1e-5`.

Phase A establishes the algebraic prerequisite: G3 gives exact dense linear and
conv parity and `1e-12` stem parity. G6 adds a smaller pure-regular structural
comparison at `C=96`: matched parameters in the typed stem, two production-form
C blocks, a register refresh, one A block, final norm, and policy/value reads
agree with `model.py` primitives to their fp64 tolerances
(`test_pure_regular_toy_path_matches_production_primitives`). Neither result
substitutes for the Phase-B state-dict and live-logit gate; both make that gate
well founded.

---

## 8. Deliberate exclusions

- **No irrep/Fourier basis.** D6 irreps are valid and may admit cheaper
  block-diagonal linear algebra, but generic shared per-coordinate GELU is not
  equivariant in those bases. A correct irrep network would need special
  nonlinearities (norms, gates, tensor products, or basis round-trips) and a
  different kernel/serve design. That is a different project; this one keeps
  permutation actions and gathers.
- **No sign channel.** The orientation character is a valid one-dimensional
  rep, but §4.1 gives the constructive GELU failure. It could be used with an
  odd or otherwise sign-aware nonlinear design, not with the locked trunk.
- **No C6 reduction.** Dropping reflections would weaken the exact symmetry
  contract of the game and would turn mirror-related positions into unrelated
  examples. All types here are representations of the full order-12 D6.
- **No quotient attention internals in Phase B.** The residual stream is where
  quotient types save width. A/L/register internals remain regular so the
  established axis-head partitions, dot products, joint bias ties, ray-side
  split, and serve folds transfer unchanged.
- **No Phase-B production or GPU work in this phase.** Phase A is CPU-only and
  creates proof/evidence artifacts. Production model, Triton, serve, training,
  and deployment changes remain blocked on an accepted GO decision.

The result is a generated family of exact full-D6 permutation fibers: cheaper
types change the number of materialized channels, while every linear, conv,
normalization, nonlinearity, attention boundary, and readout has an explicit
intertwining argument and a corresponding CPU machine check.
