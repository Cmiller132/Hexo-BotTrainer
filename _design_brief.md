# Consolidated Assessment — Seed Brief for New-Model Architecture Design

*This is the consolidated output of a teardown of the two existing Hexo models
(dense_cnn = CNN lineage; hexgt = GNN + context-transformer + PMA). It is the
shared seed for independent architecture-design agents. The goal: a NEW model,
**simpler than hexgt**, that handles the infinite board naturally (like the GNN)
while recovering the strong LOCAL reasoning + inductive bias of the CNN.*

---

## 0. The game / engine facts you must respect

- Hexo is played on an **unbounded, sparse, infinite axial-hex board**. There is
  no fixed board area. Stones sit at axial coords `(q, r)` on a hex lattice with
  **6-neighbor adjacency**. The engine only allows moves within `LEGAL_RADIUS=8`
  of existing stones, so the *live region* is sparse and grows with play.
- The game has strong **local tactical structure** (lines/connections of stones,
  threats, blocks) AND requires **whole-board / global judgement** for value
  ("am I winning?").
- Symmetry group is the **hex D6** (12 elements: 6 rotations × reflection). The
  ideal model is D6-invariant for value and D6-equivariant for policy.
- The trainer is AlphaZero/KataGo-style: self-play → per-game compact NPZ shards
  → power-law replay window → two-phase on-disk shuffle → train. **MCTS is a
  native Rust PUCT tree** (nucleus widening, forced playouts, virtual loss,
  subtree reuse) with a **thin Torch evaluator callback**. This pipeline is
  proven and representation-agnostic — a new model should REUSE it and only
  change the eval boundary (state→features→net→per-move priors + value).
- Hardware: single workstation, Ryzen 7950X + one 12 GB CUDA GPU. Self-play
  throughput matters: it is the wall-clock bottleneck of RL.

## 1. KEEP LIST (proven good — a new model should retain the property, not necessarily the mechanism)

1. **Coordinate-free / size-free move representation.** hexgt's candidate-set
   (score the live empty cells, vocabulary = current legal moves) is the correct
   answer to the infinite board. It provably never drops a legal/defensive move.
   KEEP THE PROPERTY: no fixed action plane, no arbitrary crop that can drop a move.
2. **Hex-local inductive bias.** dense_cnn's HexConv (6-neighbor receptive field)
   gives translation-equivariant local tactical pattern recognition cheaply.
   KEEP THE PROPERTY: strong, cheap local reasoning with the right adjacency.
3. **D6 symmetry handled correctly.** hexgt gets D6-invariance *by construction*
   (invariant features + permutation-equivariant ops + invariant readout) — no
   augmentation, no augmentation-poisoning risk. dense_cnn needs explicit D6
   augmentation with a fragile square-vs-hex corner fallback. PREFER by-construction.
4. **Diffuse-prior + MCTS-sharpens.** Both models have a diffuse raw policy prior
   that MCTS sharpens into decisive play. This is healthy and proven; don't fight it.
5. **Rust MCTS + thin Torch evaluator + KataGo selfplay/shuffle/training reuse.**
   Proven, fast, representation-agnostic. REUSE verbatim.
6. **65-bin distributional value + pure-arithmetic target finalization + soft-Z
   targets (λ).** Calibratable, debuggable, externally validated. KEEP.
7. **Cheap, shape-stable forward = GPU-ideal.** dense_cnn's fixed tensor shape
   made cuDNN/TRT/bucketing trivial and forward cheap (search-bound, not net-bound).
   This is a real advantage that hexgt LOST by going dynamic-shape.

## 2. DROP LIST (overbuilt / unproven / harmful — a new model should NOT include these)

- **Fixed 41×41 dense crop** (dense_cnn): drops any move >20 cells from centroid;
  hard ceiling on the infinite board. This is the #1 thing to avoid.
- **STV lookahead heads [4,12,24]** (hexgt): +287K params, fragile checkpoint/optimizer
  graft, zero demonstrated benefit.
- **opp-policy head** (hexgt): high, barely-moving loss; no ablation shows it helps.
- **Full TSS triad** (hexgt): 1-ply hardcoded tactical oracle (injection +
  hitting-set leaf override + move-selection guard); originally 2-High-severity
  unsound; ~1% wall; structurally cannot fix the real value bug. A tiny
  terminal/immediate-win-loss leaf check is all that's justified.
- **3-generation PMA varlen value readout** (hexgt): owner's own critique says
  k=2 and the `[SIDE|PMA]` skip are redundant; a custom segment-softmax kernel +
  expansion grafts for a calibration problem the readout operator can't fix.
- **v2/v3 threat-window / count-4 feature accretion** (hexgt): grew without closing
  the diagnosed "graded urgency / open-endedness" gap.
- **Heavy dynamic-shape attention plumbing** (hexgt): two attention layouts,
  precompute-layout hoisting to dodge torch.compile graph-breaks, chunked-forward
  VRAM planning. All correct but a large engineering tax that exists *only* because
  the model is dynamic-shape and transformer-based.

## 3. OPEN PROBLEMS the new architecture MUST solve

1. **Infinite / unbounded board, natively** — never drop a legal move; no arbitrary
   fixed window. (hexgt solved this; dense_cnn did not.)
2. **Strong LOCAL reasoning with a CNN-like inductive bias** — cheap, hex-aware,
   translation-equivariant local pattern recognition. (dense_cnn had this; hexgt's
   candidate-centric message passing is a weaker local reasoner.)
3. **Whole-board / GLOBAL value integration** — THE documented failure of hexgt:
   value is confidently anti-calibrated off-distribution (`v(A)+v(B)≈+0.82`, both
   players think they're winning; more search makes defense worse). Root cause is
   **global-integration-shaped, not readout-shaped**: a shallow candidate-centric
   trunk never computes cross-board threat relationships, so no pooling operator
   can recover them. A new model needs a genuine whole-board value pathway
   (e.g. KataGo-style ownership/control field is the validated lever).
4. **Prior sharpness** — neither model's raw policy head is sharp; both lean on
   search. Acceptable, but a new model should at least be measured on a fixed
   holdout and ideally drive the raw prior sharper.
5. **Forward-pass cost / self-play throughput** — hexgt is COMPUTE-BOUND (forward
   ~78% of evaluator, ~6–8 pos/s; transformer is the dominant cost). A new model
   should target a LIGHT trunk (dense_cnn was ~23 pos/s and search-bound). Throughput
   is the RL wall-clock bottleneck.
6. **Simplicity / few moving parts** — minimal heads (policy + value, justify any
   extra), one representation, no per-head graft machinery, no hardcoded oracles.
   The owner explicitly wants FEWER moving parts than hexgt.

## 4. Reference numbers (anchors)

- dense_cnn: ~2.6M params (96ch×8blk), ~23 self-play pos/s, beats SealBot best@50ms
  from ~epoch 10 (peak 92% @ ep17), fixed 41×41×13 input.
- hexgt Model 3: ~2.58M params, ~6–8 self-play pos/s (Model 2 lighter ~12),
  forward ~78% of evaluator, plateaued ~54% vs dense_cnn, value anti-calibrated.
- Constraints: BOARD coords axial; 6-neighbor hex adjacency; D6 = 12 elements;
  GPU 12 GB; single machine.

## 5. The mandate

Design ONE new architecture that:
- handles the infinite board natively (no fixed crop, never drop a legal move),
- has strong CNN-like LOCAL reasoning + the right hex inductive bias,
- has a credible WHOLE-BOARD value pathway (learn from hexgt's value failure),
- is SIMPLER than hexgt (fewer heads, fewer moving parts, lighter trunk),
- reuses the Rust-MCTS + KataGo pipeline (only the eval boundary changes),
- and is cheap enough to keep self-play throughput high.

High-level design only. Justify every head and every moving part. Genuinely
explore — search the literature for precedents.
