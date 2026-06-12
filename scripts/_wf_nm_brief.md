# New-Model Design Brief (design-competition input)

You are designing a NEW neural-network lineage for Hexo. This brief is the authoritative envelope:
owner-locked decisions are non-negotiable; Claude-proposed defaults may be refined with justification.
Everything else is yours to design. The guiding value, stated by the owner: **complexity reined in**.

## 1. The game (engine truth)

- Unbounded sparse hex board, axial coords `(q, r)` in i16. Engine: `packages/hexo_engine` (Rust, PyO3).
- Connect6-style: Opening = P0 places exactly 1 stone at forced origin (0,0); thereafter each turn = two
  single-stone placements (phases FirstStone/SecondStone). Win = 6 contiguous own stones along any of the
  3 hex axes. No draws except truncation (max_actions cap).
- Legality: empty cell within hex-distance 8 of any stone (`LEGAL_RADIUS=8`). Branching ~300–800.
- Action id = packed u32: `((q + 2^15) << 16) | (r + 2^15)` — universal across engine/Rust/Python/frontend.
- Engine tracks incremental 6-cell windows (3 axes × 6 offsets per placement) and live threats; a shared
  Rust threat-space search (TSS) exists (`packages/hexo_models/rust/src/threats_shared.rs`).

## 2. Owner-locked decisions (the envelope — do not violate)

1. **Fix the crop architecturally.** No bounded centroid crop anywhere. The model's domain is a
   stone-anchored support set: `stones ∪ full legal set ∪ 1-ring halo` (halo = cells adjacent to support
   that aren't legal/stones; they carry features but never logits).
2. **Native hexagonal.** No square-grid emulation, no masked 3×3 square kernels, no (row,col) offsets.
   The hex lattice is the primitive: 6 directions + center, axial deltas, hex distance, D6 as the symmetry group.
3. **Local reasoning like the crop convs**: direction-typed local ops mathematically equivalent in family to
   dense_cnn's hex conv (7 taps: center + 6 hex neighbors, one weight matrix per relative direction, shared everywhere).
4. **Move vocabulary = full engine legal set.** Policy logit at every legal cell. Zero coverage loss by construction.
5. **Global context = restnet-style attention** layers over the node set (pre-norm transformer block,
   multi-head, learned relative-position bias — re-keyed on hex offsets, unbounded-safe).
6. **8 carried-through summary tokens, bidirectional**: learned init vectors join the token set at the first
   attention layer and participate in every attention layer (cells ↔ tokens both directions). Value and aux
   heads read dedicated tokens; tokens have no board position (per-head learned bias scalar vs cells).
7. **Trunk depth 8 or 9 layers** (C = conv residual block, A = attention block).
8. **Heads**: per-node policy over legal nodes; per-node opp-policy (auxiliary); main value = 65-bin
   distribution read from dedicated tokens; short-term-value (STV) heads; moves-left head. **NO spatial
   ownership/win-window aux head** (owner skipped it). Main-value readout pathway must be separated from
   aux readout pathway (the heads_v3 lesson, applied natively).
9. **65-bin value targets** ([-1,1], adjacent-bin soft targets), hard-z outcomes. D6 handled by **training-time
   augmentation** (the trusted dense_cnn approach), not architectural invariance constraints.
10. **Search semantics reused, code written FROM SCRATCH — this is NOT a fork.** Do not copy-paste or fork
    dense_cnn/restnet/hexgt model code. The new lineage gets its own greenfield Python package and Rust
    code. Existing lineages are reference for semantics and contracts only. Genuinely shared infrastructure
    (hexo_engine, hexo_utils records/state_hash, hexo_train pipeline, threats_shared) may be linked against.
    The search must preserve the PROVEN semantics: batched PUCT, prior-sorted lazy edge materialization,
    nucleus widening (policy_mass 0.95 / max_children 96 / min_children 2), FPU, virtual loss, Dirichlet root
    noise, tree/subtree reuse, transposition-cached evals keyed by engine state_hash, PCR (playout cap
    randomization), policy-init openings, deterministic mix_seed streams. TSS must be toggleable (a
    `tss_enabled` config key already landed in the shared code 2026-06-12).
11. **Trusted foundation = dense_cnn.** restnet's improvements are harvestable (legal-masked policy CE,
    65-bin value, aux-head suite, fp16 evaluator + bucketed batching, attention blocks, KataGo-style replay
    window, PCR/policy-init). hexgt/hexgnn are NOT trusted sources; do not inherit their designs (you may
    independently arrive at similar mechanics if justified from first principles).

## 3. Claude-proposed defaults (refine with justification, don't discard silently)

- Interleave `C C C A C C A C A` (9 layers): 3 convs first so receptive radius 6 ≥ window span 5 before any
  attention; 3 A's because bidirectional tokens need ≥2 rounds for the hub to function; trunk ends on A so
  tokens and nodes are maximally fresh at the heads.
- Token split: 2 → main value MLP, 2 → STV+moves-left aux MLP, 4 uncommitted hub capacity.
- Node features (port of the 13 trusted planes, native): own_stone, opp_stone, empty, legal,
  phase_second_placement (const), first_stone_of_turn (one-hot cell), player_colour (const), own_recency
  (1/(1+latest_idx−placement_idx)), opp_recency, own_hot (cell in own active ≥count-3 window), opp_hot,
  hex-distance-to-nearest-stone (replaces crop-center-distance; normalize /8), opp_last_turn.
- Conv block: 2 × 7-tap direction-typed convs with norm + ReLU, post-activation residual (dense_cnn family);
  missing neighbors (outside support) contribute zeros (= conv zero-padding semantics).
- Rel-pos bias: exact learned bias per axial offset within hex radius 8 (217 entries/head), ring buckets for
  distance 9–16, one far bucket beyond; +3 learned entries for token→cell / cell→token / token→token.
- Channels 96, 4 heads, mlp_ratio 2, ~1.2M params total. BatchNorm in conv blocks (LN as fallback knob).
- Attention impl: sdpa memory-efficient path with a materialized oracle for tests (restnet's proven pattern).

## 4. Grounded contract facts (verified 2026-06-12, with file refs)

- **Evaluator boundary pattern (dense_cnn)**: Rust builds payload {inputs: zero-copy f16 buffer, shape,
  legal CSR}; Python returns {values_bytes f32×N, priors_bytes f32 positional per legal entry}; Rust zips
  positionally with action_ids it kept, validates, descending-sorts, normalizes →
  `RustEvaluation{value, priors: Vec<(PackedCoord,f32)>}` (`dense_cnn/rust/src/mcts_eval.rs:286-390,515-580`).
  The PUCT tree (`mcts_tree.rs`) consumes (action_id, prior) pairs opaquely — it is model-agnostic and crop-free.
- **Variable-length precedent**: hexgt's boundary ships CSR candidate layouts with per-graph segment ids and
  returns variable-length positional priors; same `RustEvaluation` out, same tree.
- **Eval cache**: `HashMap<StateHash, Arc<RustEvaluation>>`, key = `hexo_utils::hash_state` (pure engine
  board hash, no encoder dependence), bounded ~1M entries.
- **Continuous scheduler + PCR + policy-init** currently live only inside `dense_cnn/rust/src/mcts.rs`
  (`run_continuous`, ContinuousSlot, MoveClass, mix_seed streams). Lockstep `search` also exists. These
  SEMANTICS must be preserved by the from-scratch implementation.
- **Plugin contract** (`hexo_train`): a model package registers an entry point in group `hexo_train.models`
  exposing build_model / training_component_overrides / generate_selfplay / select_training_samples /
  train_passes / evaluate_epoch; pipeline, config TOML parsing, diagnostics, artifacts are shared
  (`packages/hexo_train/python/hexo_train/{pipeline,registry,config}.py`).
- **Data formats**: self-play writes one compact npz shard + json sidecar per game; compact rows store RAW
  FACTS (stones, placement history, legal ids, policy as action-id→weight, outcomes) — representation-
  agnostic; encoders expand to tensors at train time (precedent: two different lineages already expand the
  same shards differently). `.hxr` binary game records store raw action-id sequences. The BC corpus
  (HF `timmyburn/hexo-bootstrap-corpus`, 6,902 decisive human games ≈ 431k positions) is raw move-lists —
  any encoder can replay+re-encode (`scripts/bootstrap_dense_cnn_restnet_hf.py`).
- **Replay/training machinery** (restnet, reusable semantics): KataGo-style mtime-ordered tapered shuffle
  window (keep 300k rows, taper 0.65), policy-surprise weighting, AdamW lr 1e-3 wd 1e-4 (matrix weights
  only), AMP, grad-clip 1.0, batch 32 (64 OOMs on the 12GB RTX 4070 Ti), D6 aug per row at train time.
- **Loss weights in production**: policy 1.0, value 1.0, opp_policy 0.25, STV 0.1 (horizons [2,6,16]
  decisions, EMA targets), moves_left 0.1 (remaining decisions clamped+mapped to [-1,1], 65-bin, masked rows).
- **Perf envelope**: single 12GB RTX 4070 Ti shared by self-play inference + training. Evaluator ≈84% of
  self-play wall clock, avg GPU batch ≈54, 512 visits, active_games 128–192. Current model ~1.5M params.
  fp16 transport + bucketed batch padding are load-bearing (cuDNN re-autotune ~925ms per novel shape).
  TRT measured 2.4–2.7× at bs128/256 on the dense model (dynamic-shape models may not export — state your story).
- **Support-set scale**: 1 stone → 217 legal cells; mid-game ≈ 600–1500; long spread games up to ~3k.
  Attention is O(N²): at N≈1000 an A layer ≈ a C block; by N≈3k attention dominates. sdpa avoids score
  materialization. Batching variable-N needs explicit design (CSR flat-concat for convs is padding-free;
  attention needs per-graph layout + padding buckets).

## 5. Deliverable: a COMPLETE design document

Write your full design to the file path you were given, then return the structured abstract. Required sections:

1. **Identity** — one-paragraph design thesis (optionally a lineage name).
2. **Input representation** — support set construction (incl. empty-board/ply-0 and halo edge cases),
   node feature table (exact formulas, normalizations).
3. **Trunk** — exact ops: stem; conv block internals (norm choice and why); attention block internals
   (rel-pos bias scheme with table sizes; token mechanics); interleave with per-layer rationale.
4. **Heads & losses** — all five heads with exact shapes, target construction, masking, weights.
5. **Symmetry** — D6 augmentation mechanics for coords/directions/bias-tables.
6. **Search integration** — full evaluator payload spec (every array, dtype, layout); prior mapping;
   cache; TSS toggle; how PCR/policy-init/continuous semantics are provided from scratch
   (fresh implementation vs extraction of shared code — pick and justify).
7. **Data pipeline** — shard schema (reuse compact concept or define new), expand-time featurization,
   replay/shuffle, trainer loop deltas.
8. **Bootstrap** — BC prefit plan from the HF corpus; optional distillation from existing self-play shards.
9. **Code architecture** — greenfield package layout (Python + Rust), what links against shared infra,
   parity/oracle test strategy, build story.
10. **Perf budget** — param count by component, FLOPs/eval estimate vs dense_cnn, batching/padding plan,
    fp16/compile/TRT story, VRAM at batch 32 training and ~256-leaf inference.
11. **Milestones** — ordered implementation plan with acceptance gates (no GPU-scheduling decisions; the
    live main_4 run question is out of scope).
12. **Envelope deviations** — anything you changed from §3 defaults (or §2, only if you found a genuine
    contradiction — flag loudly) and why.

Be precise enough that an engineer could implement without re-deriving. Prefer fewer mechanisms with
sharper contracts over more mechanisms. You may read any repo file; cite file:line for claims about
existing code. Do NOT write any code or modify any file other than your assigned design doc.
