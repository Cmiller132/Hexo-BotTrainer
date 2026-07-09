# PHASE A SPEC — quotient representations: CPU-only proof, machinery, and evidence

Status: READY FOR EXECUTION. Owner: project owner; author: orchestrating agent
(2026-07-09). Executor: implementation agent ("you"). Prerequisite reading:
`docs/quotient_reps/CONTEXT.md` (ground truth for the current model), then
`docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` §0–§4 (conventions you must
match exactly).

## 0. Purpose and framing

The current trunk pays for exact D6 equivariance with the **regular
representation**: every orbit channel is replicated over 12 fiber slots, so
FLOPs and activation bytes scale with `C = 12·C_ORBIT` even though most game
concepts are plausibly mirror-symmetric (achiral), per-axis, or fully
invariant. **Quotient (coset) permutation representations** let a channel
declare a cheaper type — 6 slots (achiral), 3 slots (per-axis), 1 slot
(invariant) — while keeping *exact* full-D6 equivariance and, critically,
keeping pointwise nonlinearities legal and every existing Triton kernel
unchanged (they consume materialized dense weights and never see the tie).

Phase A produces, **on CPU only, with zero changes to any file the live
training run imports**:

1. the generalized weight-tying machinery (a *generator*, not hand-derived
   tables), proven correct three independent ways;
2. a typed toy network proven exactly D6-equivariant end to end;
3. **evidence** that the real trained net's features are substantially
   achiral/per-axis/invariant (a checkpoint + activation audit), plus a
   calibrated FLOP/byte model ranking candidate type signatures;
4. the derivation document and a results document ending in a go/no-go
   recommendation for Phase B.

Phase A deliberately does NOT touch `model.py`, serve, training, or GPU code.

## 1. Hard constraints (violating any of these fails the phase)

- **C1 — file allowlist.** You may create the new files listed in §5 and
  nothing else. You may NOT modify any existing file. In particular
  `model.py`, `equivariant.py`, `constants.py`, `register.py`,
  `inference.py`, `_triton_*.py`, `support.py`, `features.py`, everything
  under `scripts/` and `configs/`, and all existing tests are **read-only**.
  A live training run imports this tree.
- **C2 — CPU only.** No `.cuda()`, no `torch.cuda.*` calls, no Triton
  imports. Everything must pass on a machine with no GPU.
- **C3 — exactness discipline.** All correctness tests run in fp64.
  Gather-only identities must match **exactly** (atol=0); anything involving
  an averaging projection (Reynolds) uses `atol=1e-12`; end-to-end toy-net
  equivariance uses `atol=1e-9`. Fixed seeds everywhere
  (`torch.manual_seed(0)` unless a test sweeps seeds explicitly).
- **C4 — conventions are inherited, never re-invented.** D6 element indices
  0..11 are defined by `geometry.apply_d6`. Slot layout is slot-major
  (`c = slot*width_of_type + orbit_channel` per type block). The regular
  action is LEFT translation exactly as `equivariant.build_group()` produces
  it. Your generator must be *validated against* `equivariant.py`'s tables
  (G3 below), not copied from them.
- **C5 — locked decisions (do not relitigate).** Full D6 (order 12) stays;
  permutation reps only (no irrep/Fourier basis, no sign rep — see G5);
  attention internals will remain pure regular rep in Phase B (the boundary
  design); the live run is not to be disturbed.
- **C6 — every numbered goal in §3 ends in a pytest test or a script whose
  output lands in the results document.** No "verified by inspection".

## 2. The type system to implement

All types are subgroup-quotient permutation reps of D6 acting on left cosets
`G/H` (slots = cosets, action = left translation). Canonical slot ordering:
sort each coset by its minimal element index; order cosets by their minimal
element. Implement exactly these five, keyed by short name:

| name | H (subgroup) | slots | meaning |
|---|---|---|---|
| `reg` | {e} | 12 | fully chiral, orientation-sensitive (current behavior) |
| `mirror` | ⟨σ⟩, order 2 | 6 | achiral: feature ties to its mirror image |
| `point` | ⟨rot180⟩, order 2 | 6 | point-symmetric: ties to its 180°-rotation |
| `axis` | K = stab(Q-axis), order 4 | 3 | per-axis quantities (the input planes' type) |
| `triv` | G | 1 | fully D6-invariant scalars |

σ is the reflection in `K = {0, 3, 7, 10}` (elements per
`equivariant.build_group()['cosets'][0]`) that **fixes the direction
`(1, 0)`**: determine it by applying `geometry.apply_d6` to `(1, 0)` over the
two reflection candidates in K and assert exactly one fixes it; record its
index in the derivation doc. `rot180` is the element of K that maps
`(1, 0) → (−1, 0)`. Assert both derivations in a test.

A **signature** is an ordered list of `(type, multiplicity)` in the fixed
canonical order `reg, mirror, point, axis, triv`; its width is
`Σ mult · slots(type)`. The channel layout is: type blocks in canonical
order; within a type block, instance-major, slot-minor
(`channel = block_offset + instance*slots + slot`). Document this and freeze
it — Phase B inherits it.

Explicitly excluded, with a proof in the derivation doc: any non-permutation
rep (sign rep, 2-dim irreps). Reason: a pointwise nonlinearity φ commutes
with permutation matrices (`φ(Px) = Pφ(x)`) but not with general orthogonal
actions — G5 demonstrates the failure constructively.

## 3. Goals (strict, in order; each is a gate for the next)

### G1 — group foundation parity
Build the D6 tables (mult, inv, tap action on the 7 conv taps, coset data)
inside the new module from `geometry.apply_d6` alone. Test: exact equality
with every field of `equivariant.build_group()` (run under the default env,
GROUP_ORDER=12). Also verify each `rep_action(type, g)` is a valid
permutation and is a homomorphism (`action(g)∘action(h) == action(gh)`) for
all 144 pairs, all 5 types.

### G2 — hom-space dimensions, three independent ways
For all 25 ordered type pairs `(in_type, out_type)` compute the dimension of
the space of equivariant linear maps by:
(a) **orbit count** of G acting on `(out_slot, in_slot)` pairs via
`(a, b) → (π_out(g)a, π_in(g)b)` (this is the generator's basis);
(b) **double-coset count** `|H_out \ G / H_in|` by direct enumeration;
(c) **projector rank**: the Reynolds projector
`P = (1/12) Σ_g ρ_out(g) ⊗ ρ_in(g)` on the `slots_out × slots_in` matrix
space, rank via fp64 SVD (singular values > 1e-9).
All three must agree for all 25 pairs; the resulting 5×5 table goes verbatim
into the results doc. Sanity anchors that must hold: `dim(reg→reg) = 12`;
`dim(reg→T) = dim(T→reg) = slots(T)` for every T; `dim(triv→triv) = 1`.
Repeat (a) vs (c) for the **conv basis** (orbits on
`(tap, out_slot, in_slot)` with the tap action; anchors:
`dim_conv(reg→reg) = 7·144/12 = 84`, matching the current
`w_base (7, 12, ·, ·)` free-block count).

### G3 — exact reproduction of the existing machinery
With the generator specialized to pure-regular signatures:
- `typed_linear_weight` must reproduce `equivariant.gen_linear_weight`
  **exactly** (construct the explicit bijection between `wb (12, o, i)` and
  the 12 generated basis coefficients; random fp64 `wb`, both dense weights
  equal with atol=0);
- `typed_conv_weight` likewise vs `gen_conv_weight` (84 blocks, atol=0);
- the typed stem lift (Reynolds projection with ρ_out = typed action, ρ_in =
  the 25-plane input rep built from 13 scalars + 4 axis modules at plane base
  11) must reproduce `gen_stem_weight` for a pure-regular output signature
  (atol=1e-12).
This gate proves the generator subsumes the production tie.

### G4 — typed layer equivariance, property-based
For ≥50 random signatures (seeded; multiplicities 0–4 per type, at least one
nonzero), random fp64 params and inputs: verify
`f(ρ_in(g) x) == ρ_out(g) f(x)` for all 12 g, for (i) `TypedLinear`,
(ii) `TypedConv` — checked both algebraically on the generated dense weight
(the conjugation constraint) and end-to-end on a small synthetic support
(transform stone coords by `apply_d6`, rebuild the support/neighbour table
via `support.build_support`, compare node-permuted outputs — reuse the
harness pattern of `tests/test_hexfield_eq_equivariance.py`), (iii) typed
GroupAffineNorm-analogue (full-fiber stats + per-instance affine),
(iv) `typed_group_pool` (per-instance slot mean → invariance), and
(v) pointwise GELU/ReLU on typed streams.

### G5 — nonlinearity legality, positive and negative
Positive: pointwise GELU commutes with every typed action (assert over all
types/g). Negative control: construct the sign rep (ρ(reflections) = −1) and
show pointwise GELU does NOT commute (the test asserts the violation is
large, not merely nonzero). One paragraph in the derivation doc states the
general theorem: pointwise nonlinearities are exactly legal on permutation
reps.

### G6 — typed toy network, exact end-to-end equivariance
Assemble a CPU fp64 toy net using the typed machinery **plus the Phase-B
boundary design** (this is the design rehearsal, so build it exactly):
- typed stem lift from the real 25-plane input rep into a mixed signature
  (use `reg:2, mirror:2, point:1, axis:2, triv:3` → width 51; and a second,
  16-aligned one: `reg:4, mirror:4, axis:4, triv:12` → width 96);
- 2 typed conv blocks (two convs each, typed norms, LayerScale-analogue);
- 1 attention block with **regular internals**: `TypedLinear(sig → reg:K)`
  for q/k/v with K=4 (head_dim = 4·K = 16, 3 coset heads via a
  K-parameterized `head_perm`), softmax attention with a per-pair bias tied
  jointly over (row, head) exactly like `joint_bias_lut` (reuse the LUT —
  it is signature-independent), `TypedLinear(reg:K → sig)` out;
- one gated-sum register-refresh analogue (tokens as per-instance scalars,
  slot-constant within each instance) — this pins down the Phase-B token
  design;
- a policy-style head (typed conv → expand → typed_group_pool → Linear) and
  a value-style head (typed pooled read).
Test: on ≥5 random legal positions (random stone sets through
`support.build_support` + the Python oracle featurizer `features.py`),
transform stones by each of the 12 g: policy logits must permute with the
cells and value must be invariant, fp64 atol 1e-9. Also test: a pure-`reg`
signature toy net configured to mirror the real block structure must agree
with an equivalently-parameterized net built from `model.py`'s primitives on
the same inputs (this is allowed to be a smaller-width comparison; document
the construction).

### G7 — the evidence: type audit of the real trained checkpoint
Script (CLI, CPU): loads a real hexfield_eq checkpoint (operator supplies
`--checkpoint`; the live soak writes `epoch_*.pt` files — pick the latest;
the arch env required to build the net is
`scripts/prefit_env/hexfield_eq_arm4_raylayout.env` and the script must
document and assert it, e.g. by checking `arch_meta` fields after load).

For a fiber vector `v ∈ R¹²` (one orbit channel at one cell), the
**H-achiral component** is the projection
`(P_H v)[g] = (1/|H|) Σ_{h∈H} v[gh]` (average over *right* translation by H —
right-invariant functions on G are exactly functions on G/H; derive the slot
pairings from the mult table). Report, per trunk block (forward hooks on the
real net over `--positions` (default 512) real positions supplied via
`--shards` (hexfield_eq-format) or `--random-prefixes N` fallback (random
legal self-play prefixes through the engine):
- energy fractions `E_H = ‖P_H v‖² / ‖v‖²` for H ∈ {G, K, ⟨σ⟩, ⟨rot180⟩},
  averaged over cells and orbit channels (also report per-channel
  histograms/quartiles, not just means — a bimodal split "some channels
  chiral, most achiral" is the expected and most useful finding);
- the same for the token stream and the pre-`ln_final` stream;
- optional stretch: the weight-space version (each `EquivLinear.wb` is a
  function on G via `Hom(reg,reg) ≅ C[G]`; project its coefficients the same
  way).
Output: a markdown report (tables per block + a summary) written to
`docs/quotient_reps/RESULTS_PHASE_A.md`'s audit section. Include the
interpretation rule agreed with the owner: **mirror-invariant energy ≥ 70%
across most trunk depth is a strong GO for a mirror-heavy signature.**
Note: `E_G ≤ E_K ≤ E_⟨σ⟩` and `E_G ≤ E_⟨rot180⟩` must hold (nested
invariances) — assert this as an internal consistency check.

### G8 — calibrated FLOP/byte model and signature ranking
Script: closed-form per-block matmul FLOPs **and** activation bytes for an
arbitrary signature at given (B, Npad), covering every component in
CONTEXT.md §8 (both convs per C block, A/L blocks with regular-internal
width `12·K_attn`, lane refreshes, heads). Baseline: the live config
(`reg:16`, K_attn=16, C=192, CCLACCLACLA, B·Npad = 24k). Project end-to-end
throughput with the mixed-bound model
`speedup ≈ 1 / (α·(F'/F) + (1−α)·(B'/B))` with α = 0.67 (documented
provenance: measured 21 pos/s sits between the pure-compute 18 and
pure-bandwidth 27 scaling predictions from the main_11 baseline). Emit a
ranked table over a sweep of ≥12 candidate signatures including at least:
`reg:16` (baseline), `reg:8,mirror:8,axis:8,triv:8` (C=176),
`reg:4,mirror:12,axis:8,triv:12` (C=156), `reg:8,mirror:16` (C=192-wide but
cheaper-typed — include to isolate the type effect from the width effect),
`reg:4,mirror:8,axis:8,triv:8` (C=128),
and the K_attn ∈ {8, 16} variants of the top candidates. Columns: width C,
C%16 alignment flag, params, FLOPs ratio, bytes ratio, projected speedup,
head_dim_A/L legality. This table + the G7 audit jointly nominate the 2–3
Phase-B arm signatures.

### G9 (OPTIONAL stretch — attempt only after G1–G8 are green)
CPU micro-prefit A/B: tiny nets (width ≤ 96, layout `CCA`), behavior-cloning
loss on ≤20k real positions, matched-FLOP comparison of `reg`-only vs the
top mixed signature vs a `mirror`-only extreme. Bound the effort: if data
plumbing (locating/converting eq-format prefit shards, or generating
positions via the oracle featurizer) exceeds ~a day of work, skip and record
why in the results doc. A null/ambiguous result here is acceptable and does
not block the GO decision (capacity effects at toy scale are weak evidence
either way — say so in the writeup).

## 4. Required deliverables — documents

1. **`docs/quotient_reps/DERIVATION_QUOTIENT_REPS.md`** — the math writeup,
   mirroring the style and conventions of
   `DERIVATION_D6_EQUIVARIANT_ATTENTION.md`: §1 types and canonical
   layouts (incl. the σ/rot180 identification); §2 hom-space basis theorem
   (orbit ↔ double-coset ↔ projector, with the 5×5 dim table); §3 conv-tap
   version; §4 nonlinearity legality theorem + sign-rep counterexample;
   §5 typed norms/LayerScale/pool; §6 typed stem lift; §7 the Phase-B
   boundary design as rehearsed by the toy net (typed stream, regular
   attention internals at width 12·K_attn, token typing, head/policy reads)
   with every place Phase B must make a choice called out explicitly;
   §8 what was deliberately excluded and why (irreps, C6, sign rep).
2. **`docs/quotient_reps/RESULTS_PHASE_A.md`** — every measured artifact:
   G2 table, G3/G4/G5/G6 gate outcomes (test names + counts), the full G7
   audit (per-block tables, per-channel distributions, commentary), the G8
   ranking table, G9 outcome or skip rationale, and a final **GO / NO-GO
   recommendation with the 2–3 nominated Phase-B signatures** and expected
   speedups. Be honest: if the audit says the net is heavily chiral, the
   recommendation is NO-GO (or "mirror-light signature only") — that is a
   successful Phase A, not a failure.

## 5. Required deliverables — code (complete file list; nothing else)

```
packages/hexfield_eq/python/hexfield_eq/reps.py   # the only new package module
scripts/quotient_type_audit.py                    # G7
scripts/quotient_flop_model.py                    # G8
scripts/quotient_cpu_prefit_ab.py                 # G9 (only if attempted)
tests/test_hexfield_eq_reps_group.py              # G1
tests/test_hexfield_eq_reps_homdims.py            # G2
tests/test_hexfield_eq_reps_parity.py             # G3
tests/test_hexfield_eq_reps_typed_layers.py       # G4 + G5
tests/test_hexfield_eq_reps_toynet.py             # G6
docs/quotient_reps/DERIVATION_QUOTIENT_REPS.md
docs/quotient_reps/RESULTS_PHASE_A.md
```

`reps.py` constraints: imports from the package limited to `geometry` and
`constants.DIRECTIONS` (+ stdlib/torch/numpy); it must import cleanly under
ANY `HEXFIELD_EQ_*` env (no dependence on GROUP_ORDER/CHANNELS at import);
pure CPU; fully type-hinted and docstringed in the package's existing style
(compare `equivariant.py` — cite the derivation doc sections in docstrings).
Tests that compare against `equivariant.py` set the required env before
import, following the existing test suites' pattern.

## 6. Acceptance checklist (all boxes required except G9)

- [ ] G1 tables exactly equal `equivariant.build_group()`; homomorphism
      property for all 5 types × 144 pairs.
- [ ] G2 three-way dim agreement, 25 linear pairs + conv anchors (84 for
      reg→reg); table in RESULTS.
- [ ] G3 exact (atol=0 / 1e-12 stem) reproduction of production linear,
      conv, and stem weights.
- [ ] G4 ≥50 random signatures × 12 g equivariance, all typed layers.
- [ ] G5 GELU legality + sign-rep negative control.
- [ ] G6 toy net exact equivariance on real featurized positions (2
      signatures), and the pure-reg comparison against `model.py` primitives.
- [ ] G7 audit runs on the real checkpoint; report includes per-block E_H
      tables, per-channel distributions, and the nesting consistency check.
- [ ] G8 ranking table with ≥12 signatures, calibration documented.
- [ ] All new tests pass on CPU via plain pytest (document the exact
      commands for both WSL and Windows in RESULTS).
- [ ] No existing file modified (verify with `git status` — only the §5
      files may appear).
- [ ] DERIVATION + RESULTS docs complete, with the GO/NO-GO recommendation
      and nominated Phase-B signatures.
