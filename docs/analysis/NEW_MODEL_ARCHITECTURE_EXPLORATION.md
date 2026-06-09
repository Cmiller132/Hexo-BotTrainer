# New Model — Architecture Exploration

*Exploratory design study. Goal: a **new, simpler** Hexo model that handles the
infinite board naturally (like the hexgt GNN) while recovering the strong **local
reasoning / inductive bias** of the dense_cnn CNN — with **fewer moving parts**
than hexgt.*

*Produced by a multi-agent workflow: 2 analysis agents (dense_cnn, hexgt) →
consolidation → 5 independent design agents (no cross-talk) → 1 adversarial
critic → this synthesis. Analysis/design only; no model/training code was changed
and the live `hexgt_rl_main3` run was not touched. Date: 2026-06-05.*

---

## 0. TL;DR / Recommendation

- **The five independent designers converged hard.** With no cross-talk, **4 of 5
  reached the same skeleton**: *keep hexgt's size-free move vocabulary, restore a
  true hex-conv local bias, add a KataGo-style global-pool + per-cell ownership
  head for whole-board value, and delete STV / opp-policy / TSS / PMA wholesale.*
  Two designs (**SHARC** and **HexSpark**) independently landed on the *identical*
  primitive — a submanifold sparse hex-CNN. Convergence this strong is a signal
  the design space has a natural answer.

- **Recommended architecture: a Sparse Hex-CNN** (the merged SHARC/HexSpark
  design, §6.1) — true 6-neighbour hex convolution over the live-cell set
  (occupied ∪ legal-move halo), interleaved KataGo global-pooling, policy + 65-bin
  value + ownership heads, D6 by construction. It is the most faithful realization
  of the keep/drop lists: dense_cnn's locality **and** hexgt's size-freedom, minus
  everything overbuilt. **FoveaHex** (§6.2) is the recommended *hedge* — the only
  design that recovers dense_cnn's fixed-shape, GPU-ideal throughput guarantee.

- **But do not build anything yet.** The critic found two **correlated blind
  spots that could sink all five designs at once**, and both are settleable in
  ~1–2 cheap days *on the existing hexgt model, with no new architecture*:
  1. **Is a Hexo "ownership/control" target even well-defined?** Every design's
     value fix depends on it; Hexo has no Go-style territory. **Gate experiment G1.**
  2. **Is hexgt's value failure integration-shaped (what every design assumes) or
     target/data-shaped (which every design would inherit)?** **Gate experiment G2.**
  Run **G1 and G2 first** (§8). They decide whether this whole architecture
  exercise is well-founded and which design family wins.

---

## 1. Why this study exists (owner framing)

> "The current hexgt model is too complicated, overdesigned, and has too many
> moving parts — but it does have things I like. I want a new model that can
> handle the infinite board size naturally, like the GNN does, but with the same
> good local reasoning and benefits that come from the CNN. The new designs
> should be simpler and focused on high-level design."

This is correct on the evidence. The hard-won value in the codebase is the
**representation** (size-free candidate set + D6-by-construction) and the **proven
pipeline** (Rust MCTS + KataGo selfplay/shuffle/training). The accreted apparatus
(STV heads, opp-policy, the TSS triad, the 3-generation PMA value readout, v2/v3
threat features) was built to chase a value-calibration failure that the evidence
says it could not structurally fix — so it added complexity without moving the
~54% strength plateau.

---

## 2. The two existing models — condensed teardown

Full source/doc citations in the agent analyses; the load-bearing facts:

### dense_cnn (Model 1 — CNN lineage)
- **Representation:** residual HexConv CNN over a **fixed 41×41×13 dense crop**
  centered on the occupied-cell centroid. `HexConv2d` masks two 3×3 corners → an
  exact **6-neighbour hex receptive field** (translation-equivariant local bias).
  Heads: fully-conv policy (one logit/cell), 65-bin distributional value, plus
  cheap opp-policy / short-term-value aux.
- **KEEP:** the hex-local inductive bias; the Rust-MCTS + thin Torch-evaluator
  boundary; the KataGo selfplay/shuffle/training split; cheap shape-stable forward
  (~2.6 M params, ~23 pos/s, **search-bound not net-bound**); 65-bin value.
- **DROP / fatal flaw:** the **fixed crop silently drops any move >20 cells from
  the centroid** and excludes it from the policy/search contract entirely — a hard
  ceiling on an unbounded board. D6≠square-symmetry forces a fragile corner
  fallback in augmentation. Raw prior never sharpens (rescued only by 512-sim
  search).
- **Evidence:** beats SealBot best@50ms reliably from ~epoch 10 (peak 92% @ ep17).
  Board-size generalization **never tested** — the ceiling is structural.

### hexgt (Model 2/3 — GNN + context-transformer + PMA)
- **Representation:** scores a per-position **candidate set** (live empty cells,
  `candidate_radius=3`) as a typed heterograph; relational message passing ×3–4 +
  context-transformer (self-attn + candidate→context cross-attn) ×3 + PMA value
  pooling. **D6-invariant by construction** (no augmentation). ~2.13–2.58 M params.
- **KEEP (the owner's "things I like"):** **natural infinite-board handling** —
  the candidate set is coordinate-free and size-free and provably never drops a
  legal/defensive move (block-cell present 28/28 in the defense study); **D6 by
  construction**; diffuse-prior + MCTS-sharpens behaviour; full pipeline reuse.
- **DROP (overbuilt, unproven, or harmful):** STV lookahead heads [4,12,24]
  (+287 K params, fragile graft, zero demonstrated benefit); opp-policy head (high
  flat loss); the **TSS triad** (1-ply hardcoded oracle, originally 2-High-severity
  unsound, ~1% wall, can't fix the real bug); the **3-generation PMA value readout**
  (owner's own critique: k=2 + `[SIDE|PMA]` skip is redundant); v2/v3 threat-feature
  accretion; heavy dynamic-shape attention plumbing.
- **The central failure (well-documented):** value is **confidently
  anti-calibrated off-distribution** — `v(A)+v(B)≈+0.82` (both players think they
  are winning on 51% of identical boards), value never strongly negative, and
  **more search makes defense *worse*** (block never chosen at any depth; value
  rises toward +1 as visits increase). Diagnosed root cause: a shallow
  candidate-centric trunk **never computes cross-board threat relationships**, so
  no pooling/readout operator can recover them.
- **Throughput:** self-play is **compute-bound** — forward ≈78% of the evaluator,
  ~6–8 pos/s (vs ~12 for the lighter 3-layer variant, ~23 for dense_cnn); the
  transformer is 53–64% of params and the dominant cost.

### The synthesis that seeded the design phase
**Open problems a new architecture must solve:** (1) infinite board natively,
never drop a move; (2) strong CNN-like local reasoning; (3) **whole-board value
integration** (the hexgt failure); (4) prior sharpness (measure on a fixed
holdout); (5) light forward / high self-play throughput; (6) simplicity / few
moving parts. The full seed brief is reproduced from `_design_brief.md`.

---

## 3. The five candidate architectures

Each was designed by an independent agent against the same spec template, with a
distinct seed direction, and each searched the literature for precedents.

### A — SHARC · *Sparse Hex-lattice Augmented Receptive Convolution*
A **submanifold sparse CNN** (Minkowski/spconv/SparseConvNet-style) running true
6-tap hex convolutions over only the live region. The submanifold property freezes
the active set across depth, so an N-block trunk is N gather-GEMM-scatter passes
over a few hundred cells — cost ∝ live cells, not board area. Whole-board value via
**interleaved KataGo global-pool-bias blocks** (mean / size-scaled-mean / max →
FC → channelwise bias) + a **per-cell ownership/control auxiliary**. D6: p6
group-conv (HexaConv) for rotations by construction + reflection by augmentation.
*Lit:* Graham & van der Maaten 2017/2018; Choy et al. 2019 (Minkowski); HexaConv
(Hoogeboom 2018); SS-Conv (Lin 2021); KataGo (Wu 2019).

### B — HexPatchScore (HPS) · *per-candidate ego-centric patch CNN*
Scores each legal move by extracting a **fixed-size radius-3 hex patch (~37 cells)
centered on that candidate** (relative coords → translation-invariant) and running
a **shared small hex-CNN** over each patch; move vocab = candidate set. Whole-board
context from a **separate light dilated-hex global tower** producing a summary
vector `g` + ownership field; `g` injected into each patch via concat + **FiLM**;
each candidate also samples the ownership field around its cell. **Value reads only
the global tower.** D6 by p6/p6m group-conv on the patch. *Lit:* KataGo; FiLM
(Perez 2018); HexaConv/HexagDLy; dilated conv (Yu & Koltun 2016); Neurohex.

### C — FoveaHex · *multi-resolution foveated hex rings*
A **centroid-centered stack of concentric hex rings** — fine 1:1 resolution in the
fovea, geometrically coarser (pooled super-cells) outward. **Coarsen, never crop.**
The result is a **fixed-shape tensor** (recovers dense_cnn's GPU-ideal property).
HexConv per ring; the coarse rings *are* the global pathway, augmented with KataGo
global-pool + ownership. Far legal moves are still scored by gathering from their
(ring, cell) feature + a tiny micro-feature MLP. D6 via exact-hex augmentation (no
square-corner ambiguity) or optional p6 group-conv. *Lit:* log-polar / foveated
CNNs; KataGo; FPN (Lin 2017); dilated conv; HexaConv.

### D — HexSpark · *minimal subtractive sparse hex-CNN*
Essentially the same primitive as A — submanifold sparse hex-conv over live sites +
KataGo global-pool blocks + ownership head — but framed as **"the minimal model":**
keep dense_cnn locality + hexgt size-freedom, drop everything else. D6 by
weight-tying the 6-neighbour stencil into orbits. Target ~1–2 M params,
search-bound. *Lit:* submanifold sparse conv; KataGo (ownership = localized
credit-assignment); hex CNNs for Hex (Gao/Hayward/Müller); CNN-vs-GNN for Hex.

### E — HexMix · *attention-free hex-shift token mixer (MetaFormer)*
Tokens = live cells (hexgt's set). **MetaFormer/ConvMixer block:** token-mix = a
zero/low-param **6-neighbour hex gather** (a hex-conv surrogate, D6-equivariant via
orbit symmetry), channel-mix = pointwise MLP. Global reasoning via interleaved
KataGo pool-bias + a per-cell ownership head; value reads pooled summary +
ownership summary. **O(N)** vs hexgt's O(N²) attention; claims ~15–20 pos/s, ~1–2 M
params. *Lit:* MetaFormer/PoolFormer (Yu 2022); ConvMixer (Trockman 2022); Shift
(Wu 2018); gMLP; HexaConv; KataGo; Graph-Mamba (escalation path).

---

## 4. Comparison matrix

| Criterion | **A SHARC** | **B HexPatchScore** | **C FoveaHex** | **D HexSpark** | **E HexMix** |
|---|---|---|---|---|---|
| **Infinite board** | sparse live-cell set — exact, never drops | candidate set — exact | coarsen, never crop; **soft-blurs far moves** | sparse live-cell set — exact | live-cell tokens — exact |
| **Local reasoning** | true 6-tap hex conv ★ | shared patch hex-CNN ★ | HexConv at fine ring ★ | true 6-tap hex conv ★ | 6-nbr gather (conv surrogate) |
| **Global / value path** | interleaved global-pool + ownership | separate dilated tower + FiLM + ownership | coarse rings (relational geom) + pool + ownership | global-pool + ownership | pool-bias + ownership |
| **Value-fix quality** | aggregate-only coupling (risk) | **orphaned** from local apparatus | **geometry retained** in coarse cells ★ | aggregate-only (risk) | pooling = weak global mixer (risk) |
| **Heads** | policy + value + ownership | policy + value + ownership | policy + value + ownership | policy + value + ownership | policy + value + ownership |
| **D6** | p6 by-construction + refl aug | p6/p6m by-construction | exact-hex aug / opt p6 | orbit weight-tying | orbit by-construction ★ |
| **Params (target)** | ~1–2 M | ~1–2 M | ~1–2.5 M | **~1–2 M (lightest)** | ~1–2 M |
| **Shape stability** | dynamic (sparse rulebook) | dynamic (N patches) | **fixed-shape ★** | dynamic (sparse rulebook) | dynamic (ragged N) |
| **Throughput claim** | between dense_cnn & hexgt — **uncertain** | N patch fwds — **not credible late-game** | **dense_cnn-class — credible ★** | search-bound — uncertain (stack) | ~15–20 pos/s — **speed credible** |
| **Simplicity** | moderate | moderate-low (two towers) | moderate (ring plumbing) | **highest ★** | high |
| **Biggest risk** | sparse-conv stack vs torch.compile | per-candidate cost + myopia | foveal blur on far decisive moves | ownership undefined → fix evaporates | aggregate ≠ relations |
| **Critic verdict** | promising-with-caveats | weak | promising-with-caveats | weak-as-fix / good baseline | weak-to-promising |

★ = a standout strength on that row.

---

## 5. Adversarial stress-test

### 5.1 Per-design (condensed from the critic)

- **A SHARC — promising-with-caveats.** Most architecturally complete. But (1)
  submanifold conv *only* propagates along live-cell chains, so two disconnected
  stone clusters couple **only** through the global pool — the *exact* hexgt
  failure topology, just with a richer aggregate. (2) Sparse-conv libraries build
  per-forward hash-table rulebooks with custom CUDA kernels and **no documented
  torch.compile/cuDNN path**; on Windows + one 12 GB GPU + the existing Torch
  callback this could land at or below hexgt. (3) Bucketing tames the outer batch
  but not the per-graph rulebook. **Compute claim: uncertain, leaning no.**

- **B HexPatchScore — weak.** (1) Cost is `N_candidates × patch + tower`; the
  candidate set **grows with the game**, and adjacent patches re-encode overlapping
  cells, so at 150–250 candidates it plausibly **exceeds hexgt's single forward** —
  the wrong cost shape for the bottleneck. (2) **Value reads only the global
  tower**, so all the per-candidate machinery contributes nothing to the *value*
  failure it is meant to address. (3) A radius-3 patch + FiLM scalar modulation
  cannot inject far-board geometry (a defender at distance 9 along a line) — the
  brief's root cause, per candidate. **Compute claim: not credible.**

- **C FoveaHex — promising-with-caveats.** (1) Coarsening is **soft move-dropping**
  — a sharp decisive move far from the centroid gets a mushy prior MCTS may never
  expand (and the brief's own evidence is that MCTS *amplifies* miscalibration
  rather than discovering an unproposed move). (2) The single centroid is an
  **adversarially exploitable anchor** — an opponent can push the decisive region
  into the coarse zone; the "second fovea" escape hatch concedes this. (3)
  Super-cell collision disambiguation is a subtle correctness hazard. **But** it is
  the **only design whose compute claim the critic believes** (fixed shape → full
  cuDNN/compile/bucketing). Its risk is *fidelity*, which is bounded and
  measurable; not throughput or integration, which are open-ended.

- **D HexSpark — weak as a fix / attractive as a baseline.** (1) It **admits its
  own value fix is undefined** — ownership target unclear for Hexo — and being
  minimal, it is the most exposed: if ownership is ill-posed, D reduces to "global
  pool over a local trunk" = hexgt's failure with a cheaper trunk. (2) Same
  disconnected-cluster topology as A. (3) Reflection orbit-tying on an axial-hex
  stencil has handedness subtleties that can silently break equivariance.
  **Compute: uncertain (shares A's stack risk, lighter).**

- **E HexMix — weak-to-promising-with-caveats.** (1) The MetaFormer/PoolFormer
  literature it cites is **evidence against it**: pooling is a documented *weak*
  global mixer; those nets reserve attention/spatial-MLP for the global stages and
  use pooling only locally. HexMix uses a purely local mixer everywhere + aggregate
  pooling for all global reasoning = the hexgt topology again. (2) It **explicitly
  admits** a permutation-invariant pool over unanchored coordinate-free tokens may
  miss long-range *relations*; the escape hatch (SSM / global registers) is where
  the real work would live and negates the simplicity claim. **Speed credible;
  sufficiency not.**

### 5.2 Cross-cutting blind spots (the important part)

These are assumptions **most or all five designs share**, so they could all be
wrong together:

1. **All five bet the value fix on a KataGo "ownership" head — and Hexo has no
   territory.** Ownership/control is validated *for Go*. Hexo's win condition is
   connection/structure on an infinite board with no enclosure, so a per-cell
   control target is **not obviously well-defined**. If it can't be operationalized
   into a stable, D6-consistent, outcome-correlated target, **all five lose their
   primary value pathway simultaneously** and degrade to hexgt. *Highest correlated
   risk. Nobody has shown a Hexo ownership target exists.*

2. **All five assume the failure is integration-shaped, not target/data-shaped.**
   `v(A)+v(B)≈+0.82` with "more search → worse defense" is *also* the signature of
   a **value-target / off-distribution calibration** problem (soft-Z λ targets,
   self-play never visiting losing-defense states, distributional-value
   miscalibration). All five **reuse the same KataGo target-finalization verbatim**,
   so if the bug is target/data-shaped, **every design inherits it**. No design
   proposes a data/target probe.

3. **All lean on "MCTS will sharpen the prior."** But the documented failure is
   that more search made defense *worse* — search **compounded** a miscalibrated
   value. "MCTS will fix it" is a shared crutch with direct counter-evidence,
   especially for B's and C's blurred far-board moves.

4. **All assume the Rust-MCTS / Torch-callback boundary absorbs the new
   representation cheaply.** Sparse-conv rulebooks (A, D), per-candidate batching
   (B), dual-fovea collision maps (C), and varlen token sets (E) are each *more*
   eval-boundary change than "reuse verbatim, only swap the callback" implies.

---

## 6. Deep-dive on the most promising designs

The critic's analysis splits the field cleanly: **B and E are the most fragile**
(wrong cost shape / cited literature argues against the cheap version). **A/D, C**
are the contenders. A and D are the same primitive at two complexity levels, so the
real finalists are **the Sparse Hex-CNN family (A⊕D)** and **FoveaHex (C)**.

### 6.1 PRIMARY — Sparse Hex-CNN (merge of SHARC + HexSpark)

**One model:** a submanifold sparse hex-CNN over the live-cell set, with the
KataGo global-pooling + ownership value path, built at HexSpark's minimal
complexity but keeping SHARC's by-construction rotation equivariance.

**Why it's the lead candidate**
- It is the **most faithful realization of the keep/drop lists**: a literal
  6-neighbour hex conv (dense_cnn's proven local bias) on a size-free substrate
  (hexgt's proven infinite-board answer), with every overbuilt hexgt part deleted.
- **Independent convergence** (two agents, no cross-talk) is real evidence this is
  the natural design.
- Lightest principled trunk (~1–2 M params), linear in live cells, no O(N²)
  attention — directly attacks the throughput open-problem *if* the kernel stack
  cooperates.

**The two risks that must be retired, and how**
- **Disconnected-cluster coupling (the value risk).** Submanifold conv won't relate
  two far-apart clusters except through the global pool — structurally the same gap
  that sank hexgt. **Mitigations, in escalation order:** (a) increase the *frequency
  and width* of the global-pool-bias blocks; (b) add a few **strided/pooled coarse
  sparse layers** so information flows *spatially* between clusters (the sparse-conv
  literature's prescribed fix), not just through an aggregate; (c) as a last resort,
  **a small fixed number of learned "global register" tokens** that broadcast to all
  cells — sub-quadratic, far cheaper than hexgt's full attention, and the principled
  way to restore *relational* (not aggregate) global reasoning. Note (b)/(c) are how
  you keep the value fix honest if pure pooling proves insufficient. *This is the
  single most important design decision and it should be driven by G2's result.*
- **Sparse-conv stack vs the pipeline (the throughput risk).** The rulebook /
  torch.compile / Windows-GPU integration is unproven here. **Retire it with a
  micro-benchmark before committing** (§8 step 3). If it fails, fall back to C,
  which gets dense_cnn-class throughput by construction.

**Spec (minimal):** policy (per-empty-cell logit, softmax over candidates) +
65-bin distributional value (soft-Z λ, arithmetic finalization — kept verbatim) +
ownership auxiliary (training-time only; droppable at inference, so zero MCTS cost).
D6 by construction (rotation orbit-tied stencil + reflection handled by augmentation
or p6m, gated by an exact 12-element equivariance unit test — the handedness
subtlety the critic flagged must be tested, not assumed). **No** STV, opp-policy,
TSS triad, or PMA.

### 6.2 HEDGE — FoveaHex (C)

**Keep this alive as the throughput-guaranteed alternative.** It is the only design
that recovers dense_cnn's fixed-shape, cuDNN/TRT/bucket-friendly forward — the one
compute property the critic fully believed — and its coarse rings retain *relative
geometry*, so its global pathway is genuinely *relational*, not a bag-of-features
aggregate (a real edge over A/D/E on the value problem).

**Its one risk is bounded and pre-measurable:** foveal blur on a far decisive move.
That is a *single data question* — "how often does the MCTS-chosen move lie far from
the stone centroid in real late-game positions?" — answerable offline with zero
training (§8 step 4). If far-decisive moves are rare, C is arguably the
lower-overall-risk choice (its risk is bounded; A/D's stack risk is open-ended). If
they are common, C is dead and A/D is the path. **This single measurement is the
cheapest A/D-vs-C discriminator.**

---

## 7. What every candidate keeps and drops (the simplification payoff)

**Kept (all designs):** size-free candidate-set policy; true hex-local conv;
65-bin distributional value + soft-Z; Rust MCTS + KataGo selfplay/shuffle/training
pipeline (only the eval callback changes); D6 handled correctly.

**Dropped (all designs, vs hexgt):** the per-edge-type RGCN; the context-transformer
(self + cross attention); the 3-generation PMA varlen value readout; STV lookahead
heads [4,12,24]; the opp-policy head; the full TSS triad; v2/v3 threat-window
feature accretion; all dynamic-shape attention plumbing (two layouts, graph-break
hoisting, chunked-VRAM planning). The only MCTS oracle retained is a trivial
terminal / immediate-win-loss leaf check.

That is the owner's "fewer moving parts" delivered concretely: **3 heads, one
trunk, one representation, no per-head graft machinery, no hardcoded tactical
oracle.**

---

## 8. Recommended path — cheap gates first, then build

**Do these in order. Steps G1–G2 use the existing hexgt model — no new
architecture — and can invalidate the whole exercise cheaply.**

1. **G1 — Settle the ownership target (≈1 day, no GPU training, read-only).**
   Attempt to define and *label* a per-cell Hexo control/ownership target from
   terminal self-play states (final connectivity / region attribution / influence).
   Require it to be (a) stable, (b) D6-consistent, (c) correlated with game outcome.
   **If you cannot produce such a target, four of five designs' value fix is
   invalid** — surface that immediately; the design space changes.

2. **G2 — Integration-shaped vs target/data-shaped (≈1 fine-tune).** On hexgt's
   *frozen* trunk, compare (a) a deeper/global-pooled value head vs (b) re-fitting
   the value target on a balanced off-distribution holdout that includes
   losing-defense states. If (a) drives `v(A)+v(B)→~0`, the integration diagnosis
   holds → **the Sparse Hex-CNN (§6.1) and FoveaHex (§6.2) depth-of-trunk designs
   are well-founded.** If only (b) helps, the bug is **target/data-shaped, every
   architecture here inherits it**, and the priority is fixing data/targets *before*
   any rewrite.

3. **Sparse-conv throughput micro-benchmark (≈1 day, single GPU, no training).**
   Build the submanifold hex-conv forward at realistic live-cell counts through the
   existing Torch-evaluator boundary; measure pos/s vs dense_cnn (~23) and hexgt
   (~6–8) and check torch.compile/rulebook behaviour on this Windows + 12 GB GPU
   setup. **Retire the §6.1 throughput risk before committing.** If it fails, switch
   the primary to FoveaHex.

4. **Foveal-geography probe (≈hours, offline data analysis).** On existing `.hxr`
   self-play records, measure the eccentricity (ring) of the move MCTS actually
   chose. If >~95% of high-visit decisive moves sit in fovea+ring1, FoveaHex's core
   assumption holds; if not, FoveaHex is out. **This is the cheapest
   Sparse-Hex-CNN-vs-FoveaHex discriminator.**

5. **Equivariance + op-correctness unit tests (hours, no GPU).** For the chosen
   trunk: assert the exact 12-element D6 equivariance (catching the
   reflection-handedness subtlety), and parity of the sparse/foveated hex conv
   against a dense HexConv on a packed region.

6. **Offline behavioral-clone + value-calibration probe (≈1 GPU-day).** BC the
   chosen model to dense_cnn/hexgt MCTS visit policies; measure **raw prior
   sharpness on a fixed holdout** (open-problem #4, which dense_cnn never did) and
   re-run the `v(A)+v(B)` calibration test that exposed hexgt. This is the decisive
   pre-RL go/no-go.

7. **Only then: one short RL run**, reusing the Rust MCTS + KataGo pipeline verbatim
   (swap only the eval callback), head-to-head vs dense_cnn and hexgt at a fixed
   search budget; ablate ownership on/off and global-pool block count to confirm
   each earns its place.

Nothing in steps 1–6 requires an RL run or touches `runs/hexgt_rl_main3/`.

---

## 9. Honest bottom line

The owner's instinct is validated by independent convergence: the right new model
is **a hex-conv on hexgt's size-free representation, with a KataGo-style global
value path, and everything else deleted** — the **Sparse Hex-CNN of §6.1**, with
**FoveaHex** as the fixed-shape hedge. The simplification is large and concrete (3
heads, one trunk, no oracle).

The critic's contribution is the part that matters most: **the value fix that every
design depends on (ownership) may not be well-defined for Hexo, and the failure it
targets may not even be integration-shaped.** Both are cheap to settle on the
existing model. **Run G1 and G2 before building anything** — they determine whether
this elegant convergence is solving the right problem.

---

### Appendix — provenance
- Phase 1 analyses: dense_cnn and hexgt teardown agents (source + `NOTES.md` +
  `docs/analysis/*`).
- Phase 2 consolidation: `_design_brief.md` (working file).
- Phase 3 designs: 5 independent agents (SHARC, HexPatchScore, FoveaHex, HexSpark,
  HexMix), each with literature search.
- Phase 4 critique: 1 adversarial reviewer.
- Key external precedents: KataGo (Wu 2019, arXiv:1902.10565) — global pooling +
  ownership; Submanifold Sparse ConvNets (Graham & van der Maaten, arXiv:1706.01307
  / 1711.10275); Minkowski Engine (Choy 2019, arXiv:1904.08755); HexaConv
  (Hoogeboom 2018, arXiv:1803.02108); HexagDLy (arXiv:1903.01814); FiLM (Perez
  2018, arXiv:1709.07871); MetaFormer/PoolFormer (Yu 2022, arXiv:2111.11418);
  ConvMixer (Trockman 2022); dilated conv (Yu & Koltun 2016, arXiv:1511.07122); FPN
  (Lin 2017, arXiv:1612.03144).
