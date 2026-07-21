# Line-reading conv designs — mechanical description for review

Date: 2026-07-10. Status: DESIGN NOTE, no decision taken, nothing implemented.
Purpose: a self-contained, neutral description of five mechanisms by which a
cell can read its axis lines, for conceptual review in a fresh session. Each
section states the mechanism, its intended goal, and its measurable costs.
Intended-goal statements describe what a design is *for*; none of them are
claims that the design achieves it. No design below has been trained or
benchmarked; all compute figures are estimates unless marked measured.

Source files for the facts herein: `packages/hexfield_eq/python/hexfield_eq/`
(`model.py`, `_raytap.py`, `features.py`, `constants.py`, `equivariant.py`,
`support.py`), `packages/hexo_strix/python/hexo_strix/` (`graph.py`,
`model.py`), `docs/RAYTAP_RESULTS_WAVE1.md`, `configs/hexfield_eq_main_2.toml`.

---

## 0. Shared context

**Game.** Connect6-family placement game on an unbounded hex lattice (axial
coordinates). Win condition: 6 in a row along one of 3 axes Q=(1,0), R=(0,1),
QR=(1,-1). Turn structure 1-then-2. Legality: empty cell within hex-distance 8
of any stone.

**hexfield_eq substrate (main_2 / arch A5).** Per position, a variable-size
node set: legal cells (dist ≤ 4 of a stone, the model-side support radius) +
stones + a 1-cell halo, ordered [legal | stones | halo]. Wire per position:
`feats (N, 46)`, `coords (N, 2)`, `nbr (N, 6)` neighbour row indices,
`raylen (N, 12)` u8 ray lengths, node mask. Batched to Npad (quantized 64),
typical N a few hundred (walkthrough example: N=402, Npad=448).

**Trunk (A5).** Stem conv → layout `CCACCACA`: 5 conv blocks (2 convs each,
all 10 equipped with ray-tap), 3 global attention blocks over
[6 summary tokens; cells] with a relative-position bias, register-lane token
refresh after each conv block. C = 192 channels. Total 627,343 parameters.

**Equivariance constraint (applies to every hexfield design below; Strix has
no equivalent).** The trunk is exactly D6-equivariant. Channels are tiled as
12 fiber slots × C_ORBIT=16 orbit channels; group elements permute slots and
permute the 6 conv directions (rotations 3-cycle the axes; distance along a
ray is group-invariant). Consequences used below:

- Elementwise functions (σ, ReLU, tanh) and elementwise products of two
  same-representation fibers commute with the slot permutation — they are
  equivariant with no additional tying.
- Any per-distance parameter vector must be slot-constant: free shape
  `(·, C_ORBIT)`, tiled ×12 to full width at use.
- Any learned matrix applied per cell must be a tied 1×1 group convolution
  (the existing `EquivLinear`: 12 free 16×16 blocks).
- A conv with T taps stores free weights `w_base (T, 12, 16, 16)` and
  materializes the dense `(T, C, C)` weight by index-gather each forward;
  the tie divides dense parameter count by 12 exactly.

---

## 1. Baseline: hexfield_eq A5 ray-tap conv (implemented, running)

**Base operation.** Direction-typed 7-tap hex convolution: for each cell,
gather [self, 6 hex neighbours] and apply one GEMM `(7·C → C)`, weight
generated from `w_base (7, 12, 16, 16)` = 21,504 free params per conv.

**Ray-tap modification (SPEC_RAYTAP_CONV.md §2; all 10 trunk convs equipped
under `HEXFIELD_EQ_RAYTAP=both`).** The 6 direction-tap inputs are redefined.
For cell i, direction d, channel c:

    in_d(i)[c] = Σ_{k=1..5}  α[k, c] · vis · x_{i+k·d}[c]

- `x_{i+k·d}` is the trunk fiber of the cell k steps along direction d (the
  zero row when off the support).
- `α (5, C_ORBIT)` is learned, per-distance, per-orbit-channel, shared across
  the 6 directions, tiled slot-constant. Init `α = (1,0,0,0,0)` reproduces
  the plain 7-tap conv bit-for-bit (verified: test T4).
- `vis` is a hard, input-computed visibility mask ("ray lengths"): walking
  outward from i, a cell off the support stops the walk; an anti-side stone
  is included then stops it; own-side stones and empties pass through. The
  side is assigned per orbit channel: the first orbit half uses own-side
  visibility (rays truncate at opponent stones), the second half opponent-side
  visibility. Computed by the CPU featurizer and shipped as the
  `raylen (N, 12)` wire column. (Separately noted elsewhere: the same mask is
  derivable on GPU from occupancy planes; that change is orthogonal to
  everything in this document.)
- The center tap and the `(7·C → C)` GEMM are unchanged.

**Properties.**
- The weight applied to a ray cell depends on its distance (and channel),
  never on its content. Per channel, the 5 ray values collapse to 1 value by
  a fixed linear map before any nonlinearity; the block's LN/ReLU act on the
  collapsed sum.
- The sum is signed; contributions of different ray cells can cancel.
- The only content-dependence in the read is the binary visibility mask.
- Cost per conv per cell: 7·C² MACs (GEMM) + the ray gather (30 fibers) +
  elementwise α/vis. A fused Triton serve kernel (K1) loops k = 1..5 per
  direction in-kernel; a custom autograd Function (K2) recomputes the gather
  in backward to avoid materializing the (B, N, 30, C) intermediate during
  training.

**Recorded evidence (docs/RAYTAP_RESULTS_WAVE1.md, 600-step prefits, 60-game
SealBot evals — directional, not 2σ):** the arm with ray-tap convs and the
ray-attention (L) blocks removed (A5, 8 blocks) scored 0.467 ± 0.064 vs
SealBot; ray-tap on the unchanged 11-block layout (A2) 0.333 ± 0.061; no
ray-tap (A1) 0.217 ± 0.053. The L blocks were softmax attention restricted to
ray-visible keys. A5 is the shipped main_2 architecture. The feature-set
control (A0, 25-plane v1 features) was skipped; the input-feature ablation is
unmeasured.

---

## 2. Strix (HeXONet, ported in packages/hexo_strix — external reference
architecture)

An unrelated project's bot for the same game, used in this repo as an eval
opponent. Its board reading differs structurally from hexfield_eq's.

**Graph per position.**
- Nodes: stones + legal empties (radius 8), plus one dummy global node
  connected bidirectionally to all real nodes (zero edge attributes).
- Edges ("axis graph"): from every node, walk each of the 6 signed axis
  directions up to 5 steps; each visited node gets a bidirectional edge pair.
  Walk stopping: a cell absent from the node set stops the walk; from a
  stone-origin, own stones and empties pass and an enemy stone is included
  then stops; from an empty-origin, any stone is included then stops.
  Empty↔empty edges are pruned entirely (empties receive messages only from
  stones and the global node).
- Edge attributes (5 dims): axis one-hot (3), signed distance, source-player.
- Node features (11 dims): own/opp/empty one-hots, moves-remaining, centroid-
  normalized q and r, inverse distance to nearest stone, and 4 hand-computed
  threat dims (own/opp max clean-window stone count, own/opp count of axes
  whose clean count ≥ win_length−2). No window planes, no ray-length wire.

**Network.** 4 layers of GINE message passing, hidden width 128:

    m_e   = ReLU( x_src + lin(edge_attr_proj) )       (elementwise ReLU;
                                                       no weight matrix on x_src)
    agg_i = Σ_{e: dst=i} m_e                          (unnormalized sum)
    x_i'  = MLP( (1 + ε)·x_i + agg_i )                (2-layer MLP, pre-norm
                                                       residual around the block)

Heads read the concatenation of all 4 layer outputs (jumping-knowledge cat,
width 512). Policy: MLP per legal node. Value: mean-pool over stone nodes →
MLP → tanh scalar.

**Properties.**
- Content-dependence: each edge's message passes the source cell's content
  plus a learned geometry embedding through an elementwise ReLU *before* the
  sum. There is no per-edge weight matrix; cross-channel combination happens
  in the per-node MLPs between layers.
- The receiving node does not influence its incoming messages; it enters
  only after aggregation.
- Blocking is hard-coded into edge construction (the walk rules), as in
  hexfield's visibility mask; it is not learned.
- No symmetry tying of any kind; D6/translation behavior is learned from
  data (coordinates enter as centroid-normalized node features).
- Scale context: the checkpoint used as this repo's anchor
  (`checkpoint_00237000.pt`) reflects ~237k training steps of a mature
  project. In wave-1 record-only evals it scored 0.93–0.97 against the
  600-step prefit arms.

---

## 3. Design A: 31-tap dense conv (proposed, not implemented)

**Mechanism.** Remove the pre-GEMM aggregation entirely. The conv's tap set
becomes: center + 6 directions × 5 distances = 31 taps. Every (direction,
distance) position gets its own full tied weight block in the GEMM
`(31·C → C)`. The α vector is removed (a per-distance diagonal scale is a
special case of the per-distance full block). The hard visibility mask is
retained, applied to the gathered fibers exactly as today (per-side orbit
halves). The group action extends mechanically: a group element permutes the
6 directions and fixes the distance index, so `conv_gather_index` generalizes
from (7, 12, 12) to (31, 12, 12).

**Intended goal.** Positional resolution: the exact arrangement of the 30 ray
cells reaches the learned weights without any prior compression. Distance-
pattern templates (e.g. stone/gap alternations, gap location within a line)
become expressible by weights within a single conv, and the block's second
conv can aggregate the first conv's outputs along its own rays.

**What it does not change.** The read remains linear until the block
nonlinearity: no content-dependent weighting, no learned blocking, no
receiver-conditioned reading. Signed cancellation between ray-cell
contributions remains possible (now under learned rather than fixed
weights).

**Parameters.** `w_base (31, 12, 16, 16)` = 95,232 free params per conv
(vs 21,504). Across 10 equipped convs: +737,280. Model total: 627,343 →
~1,364,600 (≈2.2×).

**Compute (estimates, unmeasured).** Conv GEMM 31/7 ≈ 4.4× per equipped conv.
At S≈450 the 10 trunk conv GEMMs are roughly 55–60% of forward FLOPs, giving
≈3× forward FLOPs. End-to-end serve slowdown is expected lower than the FLOPs
ratio because (a) the 30-cell ray gather traffic is already paid by ray-tap,
(b) fixed overheads (search, featurizer, bias builds, CUDA-graph replay) do
not scale, (c) the GEMM K-dimension grows 1344 → 5952, which raises
tensor-core utilization. To be measured with `raytap_serve_throughput.py`.

**Init / warm start.** The distance-1 direction blocks can be initialized
from a trained 7-tap `w_base` (with its α k=1 diagonal folded in) and all
k≥2 blocks zero-initialized, making the conv output-equivalent to the source
checkpoint at init. The state-dict shape changes (7→31 first dim), so loading
a main_2 checkpoint requires a one-time weight-surgery shim. The new blocks
sit directly in the GEMM gradient path (no multiplicative gate in front of
them).

**Engineering requirements.** K1 fused-kernel variant with a 31-index gather;
a K2-style recompute-in-backward Function for training (the reference path
would materialize a (B, N, 31·C) gathered tensor, ~3–4 GB fp32 at the current
batch shape); the checkpoint loader shim.

---

## 4. Design B: transmittance-gated ray conv (proposed, not implemented)

**Mechanism.** Keep the 7-tap GEMM and α exactly as today; insert three
learned, multiplicative-identity components into the tap construction. For
cell i, direction d, walking k = 1..5:

    opacity        t_j   = 1 − γ_t ⊙ σ(u ⊙ x_j + c)          (elementwise;
                                                              γ_t zero-init ⇒ t ≡ 1)
    transmittance  T_k   = vis_k ⊙ Π_{j<k} t_j               (cumulative product,
                                                              near-to-far order)
    message        m_k   = α_k ⊙ x_k + β_k ⊙ ReLU(x_k + E_k) (β zero-init ⇒
                                                              today's α term only)
    reader gate    g_i   = 1 + γ_g ⊙ tanh(W_g · x_i)          (γ_g zero-init ⇒ g ≡ 1)

    tap_d(i)       = g_i ⊙ Σ_k  T_k ⊙ m_k

- `u, c, γ_t, γ_g (C_ORBIT)`, `β, E (5, C_ORBIT)`: slot-constant tiled, like α.
- `W_g`: one tied 1×1 (`EquivLinear`-form, 12 free 16×16 blocks) per conv,
  shared across the 6 directions of that conv.
- All three components are exactly identity at init (the conv is bit-for-bit
  today's ray-tap conv), and each can be disabled on a trained checkpoint by
  zeroing its gate parameter (γ_t, β, γ_g respectively).
- Equivariance: every new op is elementwise or a product of same-rep fibers,
  plus one tied 1×1; no new tying derivations are required. The reader gate
  being direction-shared is what keeps its equivariance trivial.
- The transmittance multiplies *on top of* the hard visibility mask
  (attenuate-only): the learned component can further attenuate a ray but
  cannot pass information through cells the hard rule blocks. This is a
  deliberate design choice, not a constraint; a variant replacing `vis` with
  learned-only transmittance exists but loses exact init-equivalence.

**Intended goals per component.**
- Transmittance: generalize the binary blocking rule to learned, graded,
  order-dependent attenuation (a cell's content determines how much it
  occludes what lies behind it on the ray).
- Gated message (β/E): allow per-channel, distance-shifted thresholding of
  each ray cell's contribution before the sum, so the tap can carry rectified
  (non-cancelling) sums of per-cell threshold detections. This component is
  the direct analogue of Strix's message form (§2), with a linear bypass and
  an α weighting that Strix's plain sum does not have.
- Reader gate: a bilinear (self × line) term letting the receiving cell's
  state rescale, per channel, what it reads from all its rays. Neither the
  baseline nor Designs A/Strix contain a multiplicative receiver term.

**What it does not change.** Within the gated stream, contributions are still
summed over k, so the exact positional arrangement of a line remains
compressed per channel; there are no per-(direction, distance) full weight
blocks.

**Parameters.** Per equipped conv: u+c+γ_t (48) + β+E (160) + W_g (3,072) +
γ_g (16) ≈ 3,296. Across 10 convs ≈ +33k. Model total ≈ 660k (+5%).

**Compute (estimates, unmeasured).** One extra tied C×C GEMM per conv
(≈ +14% of the conv GEMM) plus elementwise traffic on the gathered ray cells;
≈ +8–10% forward FLOPs. The cumulative product fits the K1 kernel's existing
ordered k-loop as a running register value. K2 recompute extends (all new
ops are elementwise-recomputable from x).

**Init / warm start.** Exact: a main_2 checkpoint loads (new params appear,
all zero-init) and the network function is unchanged at load.

**Known risk (stated, not weighted).** Zero-initialized multiplicative gates
can receive small early gradients and remain near-inactive; the repo has a
recorded precedent of zero-init parameters staying at init behind masks
(spec D-S28, a different mechanism). Gate-norm diagnostics during training
would be the detection method.

---

## 5. Design C: modified 31-tap (31-tap + transmittance + reader gate)
(proposed, not implemented)

**Mechanism.** Design A's 31-tap GEMM, with two of Design B's components
composed around it:

    t_j   = 1 − γ_t ⊙ σ(u ⊙ x_j + c)
    T_k   = vis_k ⊙ Π_{j<k} t_j
    x̃_k   = T_k ⊙ x_{i+k·d}            (gathered ray fiber, attenuated)
    g_i   = 1 + γ_g ⊙ tanh(W_g · x_i)
    out   = GEMM_31( [x_i, x̃_{d,k} for all d,k] ),  ray-tap portion scaled by g_i

i.e. transmittance multiplies the gathered fibers *before* the 31-tap GEMM;
the reader gate multiplies the aggregated ray contribution *after* it. The
β/E gated-message component of Design B is omitted in this composition
(rationale recorded: with per-distance weight blocks feeding the block's
ReLU, per-cell threshold detection followed by along-ray counting is
expressible within one conv block — first conv detects, second conv
aggregates — so the explicit pre-sum gate addresses a narrower gap here than
it does over the 7-tap baseline. Whether that reasoning holds empirically is
untested).

**Intended goal.** Combine Design A's positional resolution with Design B's
two mechanisms that per-position weights cannot express at any tap count:
content-dependent ordered attenuation, and receiver-conditioned reading.

**Parameters.** Per equipped conv: 95,232 (31-tap) + 48 (opacity) + 3,088
(reader gate) ≈ 98,368. Model total ≈ ~1.37M (≈2.2×).

**Compute.** Design A's cost plus Design B's elementwise/1×1 overhead;
estimate ≈ 3× forward FLOPs, end-to-end serve factor expected below that for
the reasons in §3, to be measured.

**Init / warm start.** As Design A (weight-surgery shim; k≥2 blocks zero),
with γ_t = γ_g = 0. Output-equivalent to the source checkpoint at init.
Post-hoc component ablation available by zeroing γ_t / γ_g on a trained
checkpoint.

---

## 6. Fact table

| | Baseline (A5) | Strix | A: 31-tap | B: transmittance | C: modified 31-tap |
|---|---|---|---|---|---|
| Aggregation over a ray | fixed per-distance diagonal weights, sum | per-edge biased ReLU messages, sum | none before GEMM (per-distance full weight blocks) | gated + attenuated sum | attenuated, no pre-GEMM sum |
| Content-dependence of the read | binary visibility only | sender content via message ReLU | binary visibility only | sender (gate), path (transmittance), receiver (gate) | path (transmittance), receiver (gate) |
| Order/blocking | hard rule (input-computed) | hard rule (edge construction) | hard rule | hard rule + learned graded attenuation | hard rule + learned graded attenuation |
| Receiver influence on read | none | none (post-aggregation only) | none | multiplicative channel gate | multiplicative channel gate |
| Positional resolution of the line | 1 weighted total/channel/direction | 1 summed message/channel/direction | full (30 cells individually weighted) | 1 gated total/channel/direction | full |
| D6-equivariant | yes (tied) | no | yes (tied) | yes (tied) | yes (tied) |
| Free params, model total | 627k | ~0.86M (untied, h=128×4L) | ~1.36M | ~660k | ~1.37M |
| Forward FLOPs vs baseline (est.) | 1× | n/a (different net) | ~3× | ~1.08× | ~3× |
| Exact-function warm start from main_2 | — | n/a | yes, via weight surgery | yes, direct load | yes, via weight surgery |
| New kernel work | — | n/a | 31-index K1 variant; K2-style training Function | K1 loop additions; K2 extension | both of A's + B's |
| Post-hoc component ablation | — | n/a | no (weights entangled) | yes (zero γ_t / β / γ_g) | partial (zero γ_t / γ_g) |

Strix parameter count is approximate (input/edge projections + 4 GINE layers
+ heads at hidden 128); it is included for scale only — it is a different
network on a different substrate and not directly comparable.

---

## 7. Recorded evidence a reviewer may want

1. Wave-1 ladder (600-step prefits, 60-game SealBot evals, SE ≈ 0.05–0.06;
   directional): ray-tap added strength over no-ray-tap at matched layout
   (+0.117), and removing the softmax ray-attention L blocks on top of
   ray-tap did not cost measured strength (+0.133 point estimate) while
   removing 3 blocks. One recorded *hypothesis* (not established) for the L
   result: softmax normalization discards line-multiplicity (count)
   information; alternative explanations (L-block cost, optimization,
   600-step scale) were not separated.
2. The input-feature ablation (v1 25-plane vs v2 46-plane) was never run
   (arm A0 skipped); how much of current strength the hand-computed window
   planes carry is unmeasured.
3. Strix's strength (as this repo's anchor) demonstrates that a
   sight-line-edge GNN with 4 hand threat dims and no symmetry tying plays
   this game well at ~237k training steps; it does not localize *which* of
   its design choices carry that strength.
4. Open engineering item: a deterministic training crash at
   `HEXFIELD_EQ_PAIR_BUDGET=4.0e7` with ray-tap enabled (main_2 runs at
   1.6e7); root cause unknown; any design increasing training memory
   interacts with this unknown.

## 8. Open questions for review

1. Resolution vs interaction: which unmodelled capability is more valuable
   for this game — exact line-pattern identity (A/C) or content/receiver-
   conditioned reading (B/C)? Is there evidence either way beyond intuition?
2. Is the omission of the β/E gated message in Design C sound, given the
   two-convs-per-block argument in §5, or does pre-sum rectification carry
   value the block structure does not replicate?
3. Transmittance is attenuate-only over the hard mask; is discarding the
   possibility of seeing "through" blockers (e.g. for territory reasoning)
   acceptable? (The baseline also cannot.)
4. The reader gate is shared across a conv's 6 directions for equivariance
   simplicity; is direction-resolved receiver conditioning worth the extra
   tying derivation it would require?
5. Zero-init gates: is the dead-gate risk (§4) acceptable against the value
   of exact warm starts, or should small-noise init be preferred at the cost
   of exact init-equivalence?
6. Under Design A/C, should α-style diagonal structure be retained as a
   parallel cheap path, or is it strictly subsumed by the full blocks?
7. Is ~2.2× parameters / ~3× forward FLOPs the right spend compared to the
   same budget on width (C=192→256) or depth (more C blocks) with the
   existing 7-tap ray-tap conv? (No arm for this comparison is currently
   specified.)
8. Does Strix's empty↔empty edge pruning carry information-routing value
   that none of these designs adopt, or is it an efficiency choice its
   shallow depth made cheap? (Prior discussion recorded a preference to keep
   empty↔empty communication; the question is left open here for review.)
