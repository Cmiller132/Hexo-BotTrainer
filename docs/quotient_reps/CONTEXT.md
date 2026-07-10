# CONTEXT — the hexfield_eq model, end to end

Audience: an implementation agent working on the quotient-representation project
(see `PHASE_A_CPU_PROOF_SPEC.md` and `PHASE_B_IMPLEMENTATION_SPEC.md` in this
folder). This document is the ground truth for how the current model works.
Every claim here was verified against the working tree on 2026-07-09 (branch
`main_9-fastrow-strip`). When this document and the code disagree, the code
wins — report the discrepancy, do not silently improvise.

Read order for the project: this file → `PHASE_A_CPU_PROOF_SPEC.md` →
(after Phase A acceptance) `PHASE_B_IMPLEMENTATION_SPEC.md`.

---

## 1. The game

- Two players alternate placing stones on an **unbounded** hex grid, axial
  coordinates `(q, r)`. The opening move is forced to `(0, 0)`.
- **Legality**: an empty cell within hex-distance ≤ `LEGAL_RADIUS = 8` of any
  stone (`constants.py:15`, matches the Rust engine's `legal.rs`).
- **Win**: six stones in a line along one of the **3 board axes**
  `{Q, R, QR}` (engine: `state.rs`, "six_in_line"). All tactics are therefore
  organized around **length-6 windows**: the 6-cell segments along an axis
  through a cell (`WINDOW_LEN = 6`, `RAY_REACH = WINDOW_LEN - 1 = 5`).
- The board has an exact **D6 symmetry** (6 rotations × reflection, order 12):
  rotating/reflecting a position yields an equivalent position with permuted
  moves. Under D6 the 3 axes form the S3 permutation module (rot60 3-cycles
  Q→R→QR; reflections transpose two axes).

## 2. Position representation: the support graph

A position is presented to the net as a **list of N nodes**, not a grid
(`support.py`):

- `legal` = empty cells within hex-dist ≤ `HEXFIELD_EQ_SUPPORT_RADIUS` of a
  stone (**deploy value: 4** — deliberately below the engine's legality
  radius 8; the featurizer and search only consider these cells),
- `stones` = occupied cells,
- `halo` = the distance-(radius+1) shell around `core = stones ∪ legal`.

Node order is `[ legal | stones | halo ]`, each segment ascending by packed
action id. `Support` carries `coords (N,2) int32`, `dist (N,)`,
`nbr (N,6) int32` (row-local neighbour index per direction, −1 when absent),
and `legal_count/stone_count/halo_count`. Typical mid-game N is **~150–400**.
Batches pad to `Npad` and carry a `(B, Npad)` live-node `mask`.

## 3. Input features: versioned planes with a typed D6 action

`HEXFIELD_EQ_FEATURE_VERSION` is import-time versioned and defaults to 1:

- **v1: 25 planes = 13 scalars + 4 axis modules.** The scalar planes are
  D6-trivial: occupancy/side, `dist_to_stone`
  (scaled by `DIST_SCALE = 8`, halo = 1.125), the 2 fork planes
  (`own_fork`/`opp_fork` at indices 23/24), and the other kept basics. NOTE:
  the 13 scalars are **not contiguous** — the axis block sits at indices
  11..22 (`equivariant.py:41-52`).
- Its **12 axis-indexed planes** (indices 11..22) are 4 quantities × 3 axes —
  `own_line[a]`, `opp_line[a]` (strongest own/opp count among clean length-6
  windows through the cell, /5) and `own_live[a]`, `opp_live[a]` (count of
  windows still clean for that side, /6). Under D6 these are **4 copies of the
  3-slot axis-permutation module** (a quotient rep! — the input is already
  typed; plane `11 + q*3 + a` maps to `11 + q*3 + cosp[g][a]`).
- **v2: 46 planes = 16 scalars + 10 axis modules.** The same axis block base is
  retained, with own/opp live3/live4/live5 adding six axis quantities. Forks
  move to 41/42 and three global D6-trivial planes occupy 43..45.

Features are computed in Rust (`hexo_engine`'s incremental `WindowStore`) with
an exact-parity Python oracle (`features.py`). The featurizer reads the same
`HEXFIELD_EQ_SUPPORT_RADIUS` env var as `support.py`.

## 4. The equivariant channel structure (the ×12 fiber)

All the machinery below lives in `packages/hexfield_eq/python/hexfield_eq/`
(`equivariant.py` + `model.py`), and is the subject of the quotient project.

- Trunk width `C = GROUP_ORDER × C_ORBIT` = **12 × 16 = 192** in the live run
  (`HEXFIELD_EQ_CHANNELS=192`; import-time constants in `constants.py`).
- **Layout convention (slot-major)**: channel `c = slot*C_ORBIT + a`, with
  fiber slot `slot ∈ 0..11` (a D6 element) and orbit channel `a ∈ 0..15`. The
  regular rep acts by **left multiplication of the slot label**:
  `(ρ_reg(g) v)[k] = v[regp[g][k]]` with `regp[g][k] = mult[inv[g]][k] = g⁻¹k`.
- **Group tables** (`equivariant.build_group()`, cached): element indices
  0..11 match `geometry.apply_d6` exactly (this indexing is load-bearing for
  everything). Provides `mult`, `inv`, `tapp[g][t]` (tap permutation on the 7
  conv taps: 0 = center, 1..6 = `DIRECTIONS`), `regp`, and the coset data:
  `cosets = [[0,3,7,10],[1,4,8,11],[2,5,6,9]]` — the 3 left cosets of
  `K = stab(Q-axis) = {e, rot180, g7, g10}` (an order-4 Klein subgroup);
  `cos_of[x]`, and `cosp[g][c]` (the induced action on the 3 cosets = the 3
  win-axes).

### Weight tying (all reps here are permutation reps — gathers only, no signs)

- **`EquivLinear`** (`model.py:651`): free param `wb (12, corb_out, corb_in)`;
  the dense `(C_out, C_in)` weight is materialized as the group-circulant
  `W[out-slot a, in-slot b] = wb[a⁻¹b]` (`gen_linear_weight` +
  `linear_gather_index`). Bias is slot-constant (`bias_base (corb_out,)`
  repeated 12×). Params per layer = C²/12; **compute = full dense C²**.
- **`HexNodeConv`** (`model.py:475`): 7-tap gather (B,N,7·Cin) → one GEMM. Two
  kinds: `regular` (free `w_base (7, 12, corb_out, corb_in)`, dense weight
  `W[t, a, b] = w_base[π_{a⁻¹}(t), a⁻¹b]` via `conv_gather_index`) and `stem`
  (`C_in = 25`: free `w0 (7, C, 25)` **Reynolds-projected** each
  materialization onto the equivariant subspace using `M_reg(g)` and the input
  rep matrices `ρ_in(g)` — `gen_stem_weight`, derivation §8).
- **Serve weight cache**: both modules cache the materialized dense weight
  under `no_grad`, keyed on the base param `._version` — frozen serve weights
  regenerate once, not per forward. The cache also **folds the coset head
  permutations** into the weights (`EquivLinear.set_serve_perms`): q/k/v fold
  `W[head_perm]` (bit-identical row perm), out_proj folds `W[:, head_perm]`
  (accumulation-reordered). The fold gate is *exactly*
  `not torch.is_grad_enabled()` and the owners' forwards test the same
  condition as `folded` — the two must always agree.
- **`GroupAffineNorm`** (`model.py:614`): LayerNorm statistics over the full C
  fiber (symmetric under slot permutation) then an affine **tied per orbit
  channel** (`gamma/beta (C_ORBIT,)` tiled over the 12 slots). Exposes
  expanded `(C,)` `.weight/.bias` views so the fused Triton conv+LN kernel
  consumes it like a plain LayerNorm. **`LayerScale`**: one `(C_ORBIT,)` gamma
  tiled 12×, init 1e-4.
- **Invariant readout**: `group_pool` (`equivariant.py:358`) = mean over the
  12 slots, `(…, k·C) → (…, k·C_ORBIT)`.

### Why this is expensive (the motivation for the quotient project)

Every unique learned feature (orbit channel) is materialized in **all 12
poses**; FLOPs are those of a dense C=192 net while free params are C²/12.
The quotient project keeps exact D6 equivariance but lets channels declare a
cheaper type (6-slot mirror-achiral, 3-slot per-axis, 1-slot invariant)
instead of paying 12 slots for everything.

## 5. Trunk layout and the three block types

`TRUNK_LAYOUT` env `HEXFIELD_EQ_TRUNK`; live run: **`CCLACCLACLA`**
(5 C + 3 L + 3 A; must end with 'A'). All blocks are width-C residual blocks
on the node list.

- **C = `ConvBlock`** (`model.py:742`): post-activation residual,
  **two** `HexNodeConv`s: `relu(x + LS(ln2(conv2(relu(ln1(conv1(x)))))))`,
  masked. Reach: 1 hex ring per conv.
- **A = `AttnBlock`** (`model.py:902`): pre-norm transformer block over the
  **joint sequence `[6 register tokens ; all cells]`** — the only
  unbounded-range communication. `RelPosAttention`: **3 heads, structural**
  (= the 3 left cosets of K; `head_dim = 4·C_ORBIT = 64`; import-time errors
  forbid any other head count). Channel reorder `head_perm()` (coset-grouped)
  makes the `(3, 64)` reshape land each head on one win-axis coset; at serve
  the perm is folded into cached weights. Additive relative-position bias per
  block: a free `(n_joint_classes,)` param expanded by the **jointly
  (row, head)-tied** LUT `joint_of_row_head` (`joint_bias_lut()`: union-find
  over the diagonal action `(offset-row, head) → (g·offset, g·head)`;
  237-row domain = 217 exact disk rows (radius 8) + ring/far/token buckets).
  MLP: `MLP_RATIO = 2`, GELU (pointwise ⇒ commutes with slot permutations —
  this legality is exactly what restricts the design to permutation reps).
- **L = `RayAttnBlock`** (`model.py:1010`): pre-norm ray attention over
  **cells only**. Each cell attends to ≤31 keys: itself + up to `RAY_REACH=5`
  cells along each of its 6 ray directions. **`RAY_HEADS = 6`, structural**:
  3 win-axis cosets × {own, opp} side; `head_dim_L = 2·C_ORBIT = 32`. The
  own/opp sub-head split **must ride the orbit index** (`head_perm6()`;
  any 2+2 split of the 4 K-slots is broken by some group element —
  `equivariant.py:259-285`), hence the `C_ORBIT even` import check for 'L'
  layouts. Per-L-block bias: free `(n_joint_classes, 2)` (joint class × side).
  `RAY_BLOCKERS=1` (deploy): rays truncate at anti-side stones using the
  `raylen` wire data.
- **Register lane** (`register.py`; deploy `HEXFIELD_EQ_REG_LANE=1`,
  `REG_TOK_READ=0`): after **every C and L block**, `RegisterRefresh` updates
  the 6 summary tokens by a sigmoid-gated **unnormalized fp32 SUM** over cells
  (`Σ sigmoid(q·k + gate_bias) · v · sum_scale`) — a *counting* primitive
  (softmax can't count). q/k/v/out are EquivLinears with the 3-coset head
  split; `out_proj` near-zero init (3e-3) so the lane is a no-op at step 0.
  The token stream is **carried fp32** between A blocks on half-precision
  serve (spec D-S27). `TokenRead` (cells ← tokens broadcast) exists but is
  OFF. Tokens are stored `(NUM_TOKENS=6, C_ORBIT)` and tiled slot-constant
  (i.e. trivial-rep content in a regular-rep container).

Trunk walk (`model.py:1901-1938`): C and L blocks act on cells; each A block
concatenates `[tokens; cells]`, runs, and splits back. The **pre-`ln_final`
token stream** is saved (`pre_tokens`) because it carries the lane's count
magnitudes, which `ln_final` would normalize away.

## 6. Heads

All reads are made D6-invariant before any plain `nn.Linear` (derivation §7):

- **Per-cell heads** (`policy`, train-only `opp_policy`/`soft_policy`/
  `cell_q`): tied `HexNodeConv(C→C)` → ReLU → `EquivLinear` expand
  ×`POLICY_READ_EXPAND=2` → `group_pool` → `nn.Linear` to 1 logit
  (or `VALUE_BINS` for `cell_q`). Masked to live cells.
- **Scalar heads**: a shared `inv_read = EquivLinear(C, 4·C)`
  (`INV_READ_EXPAND=4`) feeds group-pooled 64-wide read blocks. `value` reads
  all 6 tokens + masked-mean pooled cells + the pre-`ln_final` token mean
  (8 blocks → `value_reduction` → ReLU → 65-bin distributional head,
  `VALUE_BINS=65`). `stvalue_{h}` (short-term value, tokens 2,3) and
  `moves_left` (tokens 4,5) mirror it with 4 blocks each.
- Serve uses `forward_policy_value` (policy + value, moves-left on request);
  the other heads are train-only.

## 7. Serve stack (don't break it)

The live soak serves ~21 pos/s (one "position" = one root search step). The
performance stack, all env-gated (defaults set in
`scripts/_hexfield_eq_supervise_main1.sh` and mirrored in
`scripts/systemd/hexfield-eq-supervisor-1.service`):

- fp16 serve (`HEXFIELD_SERVE_HALF`) via a frozen deepcopy; value/ml tops fp32.
- Materialized-weight caches + coset-perm fold (§4) — all Triton kernels are
  **blind to the tie**: they consume dense materialized weights.
- Triton kernels: fused hex-conv and conv+LN (`_triton_conv.py`), FA2-style
  pair-bias attention for A blocks (`_triton_attn.py`), gathered ray kernel
  for L blocks (`_triton_ray.py`, `HEXFIELD_EQ_TRITON_RAY=1`; a v2 variant
  exists and is a documented negative result, default off).
- **Sync-free ray gather-index build** (sort + searchsorted coordinate join,
  zero device→host syncs) — this is what legalized `HEXFIELD_CUDA_GRAPHS=1`
  (graph capture per (B-bucket, Npad, moves-left) key).
- Serve batching: virtual batch size 96.

Import-time constraints to respect (loud errors in `constants.py`): `C % 16 == 0`;
`head_dim ∈ {16,32,64,128}` for the Triton fast path (96 permitted but slow);
heads == 3 and `head_dim == 4·C_ORBIT` under GROUP_ORDER=12; `C_ORBIT` even
for 'L' layouts; `GROUP_ORDER ∈ {1, 6, 12}` with 6 reserved/unimplemented and
**1 = the non-equivariant passthrough** (every module has a passthrough branch
— grep `self.equivariant`).

## 8. Cost model (verified against the code, C=192, layout CCLACCLACLA)

Per-cell matmul MACs, in units of C² = 36,864:

| component | count | units |
|---|---|---|
| ConvBlocks (2 convs × 7C² each) | 5 | 70 C² |
| A blocks (qkv 3 + out 1 + MLP 4) | 3 | 24 C² |
| L blocks (same) | 3 | 24 C² |
| RegisterRefresh k/v on cells (2C² each) | 8 | 16 C² |
| serve heads (policy conv + expands + inv reads) | — | ~9 C² |
| **total** | | **~143 C² ≈ 5.3M MACs/cell** |

The dense-attention quadratic term (3 A blocks × 2·N²·C) is only ~5% of the
total at N=250. **Conclusion: ~95% of serve compute is per-cell channel
matmuls scaling as N·C², and C² = (12·C_ORBIT)² is where the quotient project
attacks.** On the serve GPU (RTX 4070 Ti: ~160 fp16 tensor TFLOPS, 504 GB/s)
these GEMMs (M ≈ B·Npad ≈ 24k, K = N = 192, fp16) sit in **mixed
compute/bandwidth-bound** territory (arithmetic intensity ~100 FLOP/B vs
ridge ~317), empirically confirmed: measured throughput lands between the
pure-compute and pure-bandwidth scaling predictions. So FLOP reductions
under-deliver unless activation bytes shrink too — a mixed type signature
shrinks **both** (C drops).

## 9. Environment and process gotchas

- **All shape knobs are import-time env** (`constants.py` reads env once).
  Tests that need a specific arch must set env **before importing
  `hexfield_eq`** (existing suites use subprocess or set env at module top —
  follow their pattern, e.g. `tests/test_hexfield_eq_equivariance.py`).
- The live-run arch env is `scripts/prefit_env/hexfield_eq_arm4_raylayout.env`
  (CHANNELS=192, TRUNK=CCLACCLACLA, REG_LANE=1, REG_TOK_READ=0,
  SUPPORT_RADIUS=4). Forgetting it builds the wrong net silently
  (default C=96, CCCACCCACCA, lane off).
- **State-dict key-set discipline**: toggles (`reg_lane`, 'L' in layout)
  change the key set; non-persistent buffers are used precisely to keep keys
  stable; `arch_meta()` (`model.py:1456`) is the checkpoint's
  self-description and loaders rebuild from it. Any new arch knob must ride
  `arch_meta` and `infer_net_kwargs_from_state_dict`.
- Tests run in the WSL venv:
  `wsl -e bash -c 'cd /mnt/e/Hexo-BotTrainer-hexgt && source /root/.venvs/hexgt-build/bin/activate && export PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python:packages/hexo_train/python:packages/hexo_runner/python && pytest tests/<file> -q'`
  CPU-only work also runs under Windows Python with the same PYTHONPATH.
- A live training soak is running from this tree's launch scripts. **Do not
  modify anything the live run imports** (see the hard file lists in the
  phase specs).

## 10. File map (the parts that matter here)

```
packages/hexfield_eq/python/hexfield_eq/
  constants.py     import-time env → all shape constants
  geometry.py      apply_d6 / d6_inverse / disk_offsets / rel_bias_index
  support.py       support-set construction (N nodes)
  features.py      versioned 25/46-plane Python oracle featurizer
  equivariant.py   D6 tables, tie gathers, head perms, joint bias LUT, group_pool
  model.py         HexNodeConv/EquivLinear/norms/blocks/HexfieldNet/heads
  register.py      RegisterRefresh / TokenRead (lane)
  inference.py     serve evaluator, SERVE_HALF, CUDA graphs
  _triton_conv.py  fused conv / conv+LN serve kernels
  _triton_attn.py  A-block pair-bias serve kernel
  _triton_ray.py   L-block gathered ray kernel + sync-free index build
docs/
  DERIVATION_D6_EQUIVARIANT_ATTENTION.md   §0 conventions … §9 contract
  PLAN_D6_EQUIVARIANT_REWRITE.md           the rewrite plan (features, tiers)
  PLAN_REGISTER_LANE_RAY_ATTENTION.md      lane + ray attention design
tests/
  test_hexfield_eq_derivation.py    numpy prototype = the derivation's exit gate
  test_hexfield_eq_equivariance.py  full-net equivariance harness (reuse pattern)
  test_hexfield_eq_perm_fold.py     serve fold gate (tolerance model documented)
```

## 11. Production typed quotient fibers (Phase B as built)

The old ×12 regular fiber remains the default and rollback path. When
`HEXFIELD_EQ_TYPE_SIG` is set, the residual stream is a canonical direct sum of
the five supported permutation-module types:

```text
reg (12 slots), mirror (6), point (6), axis (3), triv (1).
```

Within each type block channels stay slot-major, instance-minor. The residual
width is `C = Σ slots(type)·multiplicity`; `HEXFIELD_EQ_ATTN_ORBIT=K_attn`
independently selects a pure-regular attention interior of width `12·K_attn`.
Every A block, L block, and register refresh therefore crosses the same typed
boundary:

```text
sig --TypedLinear(q/k/v)--> reg:K_attn
    --existing attention math--> reg:K_attn
    --TypedLinear(out)--> sig.
```

Convs use the Phase-A `(tap,out_slot,in_slot)` orbit bases. MLPs map
`sig -> 2·sig -> sig`; tokens, norm affines, biases, and LayerScale are stored
per type instance and expanded across quotient slots. Invariant heads pool the
slots of each instance before any unconstrained `nn.Linear`.

The materialized dense layouts and serve contracts do not change. Triton sees
ordinary `(7,Cin,Cout)` conv weights, `(Cout,Cin)` linear weights, and expanded
`(C,)` norm affines. Head permutation folding remains confined to the regular
side of the attention boundary and uses the same no-grad cache gate.

Pure `reg:K` modules specialize to the historical `wb`, `w_base`, and
`bias_base` parameter names/shapes. Phase B's D8 gate loads the live pure-regular
checkpoint and reproduces its policy/value/moves-left logits exactly. Mixed
checkpoints record canonical `type_sig` and `attn_orbit` in `arch_meta`; missing
mixed metadata is rejected rather than guessed.

The three owner-accepted production candidates are B1 (`C=160,K=8`) and B2/B3
(`C=128,K=16/8`). Training and deployment remain deferred; see
`RESULTS_PHASE_B.md` for gates and the later ray-tap merge boundary.
