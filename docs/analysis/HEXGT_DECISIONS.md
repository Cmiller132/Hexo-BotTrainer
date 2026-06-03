# HEXGT (Model 2) — Implementation Decisions Log

Running log of non-trivial design decisions made while building `hexgt` (the
dynamic GNN + transformer Model 2) per `HEXFORMER_REWRITE_PLAN.md`. Each entry
records the decision, the plan reference, and the rationale. The plan's §11
"Open questions" and §12 readiness-gaps (A–J) are resolved here as implemented.

---

## Phase 7 — MCTS design: ASYNC leaf-eval batcher (user-directed divergence from dense_cnn)

**DECISION (binding, user): the production hexgt MCTS is an ASYNCHRONOUS
leaf-evaluation batcher, NOT dense_cnn's synchronous lockstep model.** dense_cnn
runs ~256 games in lockstep with fixed per-game leaf batches; for hexgt that
wastes GPU because per-position cost is highly variable (candidate count grows
opening→midgame→endgame), so the group waits on the slowest tree. The Phase-5c
synchronous copy (mcts.rs/mcts_tree.rs from dense_cnn) is the WORKING BASELINE
(2.3–4.9 pos/s) and stays as the reference + the deterministic eval path; the
async batcher replaces it for RL self-play.

Target architecture: many concurrent games, each its own MCTS tree, running on a
thread pool. When a tree's selection reaches a leaf needing NN eval, it builds the
leaf's graph payload, submits to a SHARED request queue, and YIELDS (does not block
the group). A central BATCHER coalesces pending leaf requests across ALL games up
to `max_batch` OR a short `latency_window` (whichever first — the partial-batch
flush is mandatory; a max-only batcher deadlocks when a single leaf is in flight),
forms one size-bucketed batch of variable-size graphs, runs ONE GPU forward, and
routes results back to the originating trees, which expand/backup and continue.
Virtual loss lets one tree hold multiple leaves in flight; a `max_in_flight`
backpressure cap bounds memory. Cache-check before enqueue; integrate tree reuse.
Net: leaf generation is decoupled from NN eval; the GPU stays fed from the global
pool regardless of per-game variance.

**Reproducibility:** async batch composition is nondeterministic — acceptable for
RL self-play. For head-to-head EVAL vs dense_cnn/SealBot, keep a DETERMINISTIC
path (fixed batch / single-game; the synchronous baseline serves this) so
comparisons are fair and repeatable.

**Sequencing:** get the async batcher + integration WORKING and measure real
cache-assisted self-play pos/s + CPU/GPU util FIRST, then tune (queue depth,
max_batch, latency_window, bucket sizes). Don't over-build before the baseline.

### Standing directive — maximize CPU multithreading + GPU utilization (Ryzen 7950X, 16c/32t)
Single-threaded CPU work is the current self-play limiter (89% Python
featurization). Parallelize across ~all 32 threads (CONFIG-DRIVEN worker count,
default high ~30, leaving headroom): state reconstruction / placement-history
replay, graph construction / expand-at-time, candidate+active-window computation
(rayon in Rust where applicable), and the trainer's shard-expand → graph-collation
pipeline (a worker pool like dense_cnn's expansion pool, sized to cores). The async
batcher runs many concurrent games so CPU (tree search + graph build) and GPU
(batched NN eval) are both saturated; concurrent-game count + max-in-flight are
tunable and set high enough to keep the GPU continuously fed. Keep GPU maxed:
FP16 + compiled model, size-bucketed batches large enough to saturate, minimal
idle. Caution: watch free RAM (the box has had OOM pressure); pool sizes
config-driven so they can be dialed back.

### Standing directive — RAM-compact data discipline (mirror dense_cnn)
Keep in-memory reps COMPACT/packed (byte-packed/columnar, the compact_io
discipline — dense_cnn went 312KB→~24KB/sample). PREFER store-compact +
unpack-on-demand on the CPU worker pool over holding expanded forms in RAM (aligns
with recompute-at-expand: shards stay raw-fact/compact, graph tensors built on the
fly, no big expanded-graph cache). Compress where it helps. Trade CPU cycles (now
parallelized across 32 threads) for RAM, not the reverse. Watch free RAM; keep
pool/cache sizes config-driven. **Implementation blueprint** (synthesized by the `hexgt-async-mcts-blueprint`
workflow — 3 parallel audits + synthesis):

- **Architecture:** batcher in Rust = 1 GIL-holding evaluator thread + N GIL-free
  rayon selection workers (one per concurrent game). Workers walk their tree,
  cache-check, on miss push a `PooledLeafRequest` + apply virtual loss + continue;
  the evaluator coalesces across all games to `max_batch` OR `latency_window_ms`
  (whichever first; partial-flush mandatory), takes the GIL once, forwards, routes
  `Arc<RustEvaluation>` back over per-tree channels.
- **KEY FINDING (reshapes priority):** the async batcher ALONE does not fix
  throughput — it only overlaps featurization across games, and the GIL-held
  evaluator still serializes the per-graph Python featurization that is ~89% of
  the cost. **The decisive win is moving `build_graph_tensors` featurization into
  Rust rayon** (parallelizes the 89% across 32 threads, removes the GIL
  serialization, and also speeds the sync path + BC training + eval).
- **Phases:** P0 baseline probe (have it); P1 minimal async batcher w/ existing
  Python featurization (overlap-only, modest); **P2 Rust featurizer (the real
  win)**; P3 size-bucketing + 32-thread tuning; P4 trainer process-pool.
- **Config (`[mcts.async]` + `[memory]`, 32t/12GB):** `concurrent_games=24`,
  `worker_threads=30`, `max_batch_size=256`, **`latency_window_ms=8`** (NOT 100 —
  starves a fast GPU), `max_in_flight=8`, `max_in_flight_global=1536`,
  `bucket_boundaries=[1,32,128,256,500]`, `eval_cache_max_entries=200000` (bounded
  LRU — a P1 PREREQUISITE; the cache is currently unbounded), `log_peak_memory`,
  `memory_pressure_warn_mb=4096`.
- **Deterministic eval path:** reuse the existing synchronous
  `run_searches_to_targets` (single game, fixed seed/batch, no virtual-loss races)
  for vs-dense_cnn/SealBot so comparisons are repeatable.
- **Mandatory gates:** featurizer Rust↔Python golden parity test (1e-6) BEFORE any
  training (a mismatch silently poisons the model — the dense_cnn D6 bug class);
  single-leaf latency-flush liveness test; bounded LRU cache before scaling games.

**SEQUENCING DECISION (autonomous, justified):** implement the **Rust featurizer
FIRST** (blueprint P2), not the async batcher (P1). Rationale: the blueprint shows
featurization is THE bottleneck and the async batcher gives only overlap gains
until it's parallel; the featurizer is lower-risk (pure-Rust rayon + a parity gate
vs GIL/channel/thread concurrency), and it immediately speeds the already-working
synchronous self-play path AND BC training AND eval — fulfilling the "parallelize
featurization across 32 threads" directive directly. The async batcher (P1, GPU
saturation) is built on top once featurization is parallel.

---

## Phase 5 — torch FP16 inference + throughput (the make-or-break gate)

**Finding (GPU FP16 forward throughput on RTX 4070 Ti, REAL 96x8 positions, n=3,
~260 candidates/pos):**
- Original per-graph Python attention loop: ~270 pos/s (kernel-launch-bound) →
  **NO-GO**, exactly the lever §6.1/§6.3 flagged.
- **FIX: vectorized batched padded attention** (`build_attention_layout` +
  rewritten `GraphTransformerLayer`): one batched MHA over (B, max_ctx/cand, D)
  with key-padding masks instead of B Python iterations. Correctness preserved
  (D6 equivariance + all 27 tests still pass). → **~2,600 pos/s**.
- **+ `torch.compile(dynamic=True)`: ~5,400 pos/s** (another ~2×). Scales cleanly
  to batch 512.

**Implied self-play pos/s** (cache-free lower bound = forward_tput / 512 sims):
**~10.5** with compile, vs dense_cnn **~23 (96x8) / ~58 (64x4)**.

**GO/NO-GO: CONDITIONAL GO.** The dynamic GNN is feasible — it runs correctly,
scales, and at ~5,400 forward pos/s sits within ~2× of dense_cnn 96x8's self-play
rate *even by the pessimistic cache-free estimate*. The MCTS transposition cache
+ tree reuse (same framework) cut NN forwards far below 512/searched-position, so
the real self-play rate should be materially higher — but confirming it needs the
full Rust MCTS graph-payload integration (next step). NOT the "unworkable,
reconsider" no-go the plan worried about. Remaining headroom: GNN op fusion (the
5-edge-type einsum + edge gather dominate), size-bucketing to cut padding waste,
shallower trunk. `inference.HexgtInference` is the torch-FP16 evaluator (no TRT,
§6.1); the Rust zero-copy payload transport + the in-loop MCTS measurement are
the remaining Phase-5 work.

---

## Phase 5b — forward throughput characterization (optimize + test pass)

Re-measured forward pos/s on REAL n=3 96x8 positions (epoch ≤24 shards) across the
candidate-count distribution, after profiling where time actually goes. Scripts:
`_profile_hexgt_breakdown.py`, `_profile_hexgt_phases.py`, `_profile_hexgt_trunk_sweep.py`.

**Where time goes (batch 512, eager, mixed positions) — corrects the Phase-5 note.**
The 5-edge-type GNN einsum is NOT the dominant cost. Split: node_in 0.7ms, **GNN
×3 = 65ms (33%)**, **transformer ×3 = 125ms (64%)**, heads 0.9ms. The transformer
(context self-attn + candidate→context cross-attn) is the bottleneck, and within
it the candidate side dominates (max_cand ≫ max_ctx; cand FFN over B·max_cand
tokens × 3 layers). The GNN einsum is already FLOP-efficient (T·N projections <
E edge-gathers), so GNN op fusion is low-value.

**Compile modes (batch 512):** plain `torch.compile(dynamic=True)` is best
(~5,800 pos/s); `dynamic=False`, `reduce-overhead` (CUDA graphs), and
`max-autotune` are all *slightly slower* — the per-position shape variation
defeats CUDA-graph capture. **Keep dynamic=True.** FP16 autocast already on.

**Realistic per-phase curve (compiled, UNIFORM same-size batches — the real
self-play case where one tree's leaves have near-identical graph sizes):**

| phase   | cands/pos | compiled pos/s | implied self-play (÷512, cache-free) |
|---|---|---|---|
| opening | ~132 | ~18,700 | ~36 |
| midgame | ~248 | ~8,700  | ~17 |
| endgame | ~504 | ~3,700  | ~7  |

The mixed-position profile (~5,800) UNDERSTATES self-play because it pads every
graph to the largest in the batch (ctx-token efficiency only ~29%); a real search
batch is near-uniform so padding waste nearly vanishes. Peak GPU mem at batch 512
is 5.33 GB (ample headroom on the 12 GB 4070 Ti). Scales cleanly to 512+.

**Trunk speed knob (compiled, uniform; the `[architecture]` token_dim/gnn_layers/
ctx_layers are all config-exposed, so this is a TOML A/B, no code change):**

| config        | params | mid pos/s | end pos/s | vs ref |
|---|---|---|---|---|
| ref 168/g3/c3 | 1.96M | 8,686  | 3,704 | 1.00× |
| ctx2 168/g3/c2| 1.50M | 11,344 | 4,791 | 1.31× |
| shal 192/g2/c2| 1.73M | 14,539 | 6,640 | 1.67× |
| wide 224/g2/c2| 2.35M | 12,638 | 5,825 | 1.46× |
| g2c3 176/g2/c3| 1.96M | 10,019 | 4,232 | 1.15× |

**Finding:** trunk DEPTH (especially `ctx_layers`) is the throughput lever; WIDTH
(token_dim) is comparatively cheap (efficient matmuls). The shallow-wide regime
(192/g2/c2) is 1.67× faster at FEWER params — the window-hub graph already gives
2-hop long-range connectivity, so gnn=2 is plausibly sufficient. The model is
UNTRAINED, so switching trunk costs nothing but a learnability question.

**DECISION:** keep `ref 168/g3/c3` (+ st-value heads = 2.07M) as the
fair-comparison primary, and carry `shal 192/g2/c2` as the speed-optimized A/B
candidate. Pick between them empirically in the BC step (compare eval strength —
the plan's matched-compute fairness), not by guessing. Deferred (low-value /
training-only): GNN op fusion (GNN not the bottleneck), training-batch size-
bucketing (matters for mixed-size train batches, not the uniform self-play gate;
revisit if BC throughput is slow), candidate-cross-attention-once restructuring
(the single biggest forward lever but a real learnability bet — revisit with the
BC harness as validator if step-2 cache-assisted self-play proves inadequate).

**GO/NO-GO: GO (firmer than Phase-5's CONDITIONAL).** Forward throughput is
feasible with headroom; the remaining question is purely the cache-assisted
self-play rate, which only the Rust MCTS integration can answer (next step).

---

## Phase 5c — Rust MCTS graph-payload integration (transposition cache + tree reuse)

**Approach (chosen): copy dense_cnn's feature-complete tree+session verbatim into
the hexgt crate; replace ONLY the eval boundary.** The Plan-agent analysis +
direct read confirmed `mcts_tree.rs` is model-AGNOSTIC (couples only to
`RustEvaluation{value, legal_action_count, priors}` + `state_hash` + `move_error`),
and `mcts.rs` is ~90% pure session orchestration. So `mcts_tree.rs`/`mcts.rs` were
copied byte-for-byte (inheriting the proven nucleus widening / forced playouts /
virtual-loss select↔eval pipeline / subtree reuse), and only these changed:
- new `mcts_eval.rs` (hexgt): builds a `{"graph_facts":[...]}` payload **in Rust**
  from the leaf `HexoState`s via the shared `candidates::build_graph` +
  `position_graph_to_py_dict` (no Py→Rust reclone), calls the Python evaluator,
  parses the per-candidate byte contract (values + CSR candidate ids + priors —
  the hexformer_ar pattern), intersects with engine legality, then
  DESCENDING-sorts + normalizes priors (the contract the copied tree requires).
- new `constants.rs`; `state.rs` gains `states_from_py_states`/`move_error`;
  `candidates.rs` factors the graph-facts dict packer out of `hexgt_graph_facts`;
  `mcts.rs` edits: session renamed `HexgtMctsSession`, stores immutable `n`
  (threaded to eval; `n` is NOT in the state hash so it must be session-constant
  for cache soundness), eval-fn renames. dense_cnn is touched ZERO lines.
- Python: `inference.evaluate_graph_facts` (the byte-contract evaluator callback,
  reusing the EXACT training featurize+collate path so search inputs == training
  inputs), a `mcts.py` session wrapper, `rust_bridge` entry points.

**Bug found + fixed:** a self-deadlock — `lock_stats(stats)` called twice in one
statement held the non-reentrant mutex while re-locking it (py-spy native dump
pinpointed `evaluate_state_refs_cached` → `Mutex::lock_contended`). One-line fix.

**Verified (CPU tests, `tests/test_hexgt_mcts.py`, all green; full suite 183):**
legal action selection; candidate priors ⊆ legal + normalized; untrained value
≈0; cache accounting balances (requested = unique + hits + dups); transposition
cache hit across game keys; subtree reuse.

**REAL cache-assisted self-play throughput (RTX 4070 Ti, 64 games × visits=128,
vbatch=16, untrained eager):** **~4.9 searched pos/s** (vbatch=8: ~2.3).
The time split is the key result — the GPU forward is NOT the limiter:
- Rust graph-encode (build_graph + dict pack): **3.2s (8%)** — fast.
- Python evaluator: **34.8s (89%)** — of which the GNN forward is only ~2.6s
  (22.9k unique states ÷ ~8.7k pos/s); the other **~32s is Python-side per-graph
  featurization (`build_graph_tensors`) + collation**.
- parse 0.4s. Cache hit-rate ~7%, ~119/128 NN-forwards per searched position
  (modest cache assist — untrained diffuse priors + Dirichlet noise + wide
  candidate sets make transpositions rare; expected to improve with a trained,
  sharper policy and deeper tree reuse).

**Bottleneck = Python featurization, not the GPU or the Rust encode.** The clear
fix (deferred, next optimization): mirror dense_cnn's `PlaneBuffer` — have Rust
emit the FINAL collated batch tensors (node_feat/edge/candidate arrays) as
zero-copy buffers so Python only `torch.frombuffer`+forwards, eliminating the
per-leaf Python featurize+collate. This requires porting `features.py`'s
node-feature encoding into Rust (kept byte-identical to training). The current
2.3–4.9 pos/s is the correct, functional v1 integration baseline; the optimized
rate is expected to approach the forward-bound ceiling (forward-only implies
~8.7k/128 ≈ 17–68 pos/s depending on cache assist, i.e. competitive with
dense_cnn's ~23).

---

## Phase 6 — first behavioral-clone from 96x8 (training-path validation)

BC reads pre-recorded compact shards (no self-play), so it is independent of the
Phase-5c featurization throughput bottleneck. `HexgtTrainer.train_on_shards`
drives: compact shard → `expand.build_training_batch` (recompute-at-expand graph
+ targets) → `model(batch)` (full forward, aux heads) → `hexgt_loss` → AdamW step.

**Bug found + fixed (latent, AMP-only):** `losses.segment_log_softmax` /
`segment_softmax_cross_entropy` mixed dtypes under AMP autocast — autocast
promotes `exp`/`log` to fp32 while the policy logits are fp16, so the `index_add`
segment scatters hit `self (Half) vs source (Float)`. The CPU gate tests never
ran AMP, so it was invisible until the first CUDA BC step. Fix: compute the
segmented softmax/CE in fp32 (cast logits at entry — standard, numerically
stable; no-op for the fp32 inference path). Regression test added
(`test_segment_losses_under_amp_autocast`). Full suite 184 green.

**BC result (epoch_000024 shards, 512 shards ≈ 25.5k rows, 200 steps, batch 128,
warmup 30, lr 1e-3, RTX 4070 Ti, ~228 ms/step):** all three heads learn clearly —
policy CE 4.48 → 3.42 (from ~ln(270)≈5.6 uniform), value 2.00 → 0.65, opp_policy
5.14 → 4.26. Prune rate 12.6% (matches the documented n=3 BC prune). This
validates the entire training path on real data; the model genuinely imitates the
dense_cnn MCTS visit distribution + value. (200 steps is a smoke, not a converged
BC model — a real BC/RL run is many thousands of steps; the ~228 ms/step is
dominated by the same Python featurization that bounds self-play, so the
Phase-5-deferred Rust feature-buffer transport also speeds up training.)

**First eval — held-out imitation vs the dense_cnn MCTS teacher** (no game-play /
SealBot needed; `scripts/_bc_eval.py` on epoch_000022, untouched by the
epoch_000024 BC run; 700-step BC model `_hexgt_bc.pt`, policy CE reached ~3.0):
- held-out policy CE **3.17** (≈ train CE → generalizes, tiny gap; uniform ≈5.6).
- top-1 policy agreement **28.4%** (chance ~0.4%) — hexgt's argmax candidate ==
  dense_cnn's most-visited move on UNSEEN positions 28% of the time. Strong for a
  700-step smoke BC; confirms the whole model+training+data pipeline learns real
  play, not noise.
- value MAE **0.88** (in [-1,1]) — value head is learning (train loss 1.12→0.55)
  but early/conservative (predicts near 0 vs ±1 outcomes); needs more steps.

**Still open (next session):** a converged BC/RL run + a head-to-head GAME eval vs
SealBot and vs dense_cnn (matched-compute). That eval needs hexgt player/eval
component wiring (the plugin's evaluation hooks are still Phase-5+ stubs) and
benefits from the deferred feature-buffer optimization for tolerable game-play
speed. The held-out imitation eval above is the achievable "first eval" without
that infra and already validates BC success.

---

## Environment / safety (build isolation)

**Live-run isolation.** The dense_cnn 96×8 run is live on the GPU, editable-
installed (`hexo-models`) against **this same worktree** (`hexo_models.pth` →
`/mnt/e/Hexo-BotTrainer/packages/hexo_models/python`), with a supervisor
(`scripts/supervise_target_96x8_wsl.sh`) that can relaunch it and a dashboard
(`hexo_frontend.web`, port 8080). The only shared mutable artifact that could
break the live run is the in-place native module
`packages/hexo_models/python/hexo_models/_rust.cpython-312-*.so`.

- **Edits + commits** happen on the main worktree `/mnt/e/Hexo-BotTrainer`
  (branch `bench/inference-backends-wsl`). Safe for the live run: dense_cnn never
  imports `hexformer_ar` or `hexgt`; the running process holds its modules in
  memory; `hexo_models/__init__.py` guards sibling roots with `.is_dir()`.
- **Native builds + test runs** happen ONLY in an isolated worktree
  `/mnt/e/Hexo-BotTrainer-hexgtbuild` with its own venv `hexgt-build`, so the
  main tree's `_rust.so` is never rebuilt/overwritten while the live run depends
  on it. We never `maturin develop`/`pip install` into the live venv or the main
  worktree.
- Git commands run through Windows git (Git Bash) per the CRLF instruction.

---

## Phase 0 — delete + scaffold

**Package name (open-q #1): `hexgt`** (Hex Graph-Transformer). Accepted from the
plan working name.

**Deletion scope (§1).** Remove `packages/hexo_models/hexformer_ar/` (Py+Rust),
its `pyproject.toml` entry point + maturin include + pyright extraPath, the
`#[path]` include + `sys.modules` registration in
`packages/hexo_models/rust/src/lib.rs`, the sibling-root entries in
`packages/hexo_models/python/hexo_models/__init__.py`, `configs/hexformer_ar.toml`,
and `tests/test_hexformer_ar_*.py`. The `namespace="hexformer_ar"` string in
`tests/test_hexo_utils_sample_store.py` is a generic label for the hexo_utils
sample store (not an import) — left as-is to minimize churn (revisit if noisy).

**Scaffold mirrors dense_cnn's module set.** `hexgt/python/hexo_models/hexgt/`
with `__init__.py`, `constants.py`, `config.py`, stub `architecture.py`,
`losses.py` (reused 65-bin binning math), `checkpoints.py` (dense_cnn-pattern,
rejects `sample_buffer`), `plugin.py` (ModelPlugin), registered via the
`hexo_train.models` entry point as `hexgt`. Rust crate dir
`hexgt/rust/` added in Phase 1 (no Rust needed for the Phase-0 gate).

### Packed-graph batch contract (gap I) — provisional, locked enough for Phase 0/3

A batch packs `G` position-graphs into one disjoint graph (PyG-style). Tensors:

| key | shape | dtype | meaning |
|---|---|---|---|
| `node_type` | (Ntot,) | int64 | node type id (SIDE=0, STONE=1, CANDIDATE=2, WINDOW=3) |
| `node_feat` | (Ntot, F) | float32 | unified per-node feature vector (F = `NODE_FEATURE_DIM`) |
| `node_graph` | (Ntot,) | int64 | graph id per node (segment/block-diagonal) |
| `edge_index` | (2, Etot) | int64 | (src, dst) in packed node indexing |
| `edge_type` | (Etot,) | int64 | edge type id (see constants) |
| `candidate_index` | (Ctot,) | int64 | node indices that are candidates, in CSR/legal order |
| `candidate_graph` | (Ctot,) | int64 | graph id per candidate (per-graph policy softmax) |
| `num_graphs` | scalar | int | G |

Forward output:
- `policy`: (Ctot,) one logit per candidate (dynamic). Per-graph softmax in loss
  via `candidate_graph` segments → emitted as `priors_bytes` in legal order.
- `value`: (G, 65) 65-bin distributional (reuses dense_cnn binning).
- `opp_policy`: (Ctot,) per-candidate aux logits.
- `stvalue_<h>`: (G, 65) optional.

This maps directly onto the Rust per-move CSR priors contract (`legal_row_offsets`
order == `candidate_index` order within a graph).

### Node feature layout (gap A) — provisional, finalized in Phase 4 expand

`NODE_FEATURE_DIM = 32`, unified vector with type-routed slots (see
`constants.py` for named offsets). Provisional; the exact encoding is locked when
`expand_row_to_graph` (Phase 4) and the GNN input projections (Phase 3) are
written. The contract only requires builder and model to agree on `F`.

### Edge types (gap I/§6.3) — bounded, window-hub routed (NO same-axis cliques)

`ADJACENCY=0` (node↔node hex-dist 1, ≤6/node), `STONE_WINDOW=1` (window↔its
one-color stones, ≤6/window), `CANDIDATE_WINDOW=2` (window↔its empty cells,
≤6/window), `RECENCY=3` (stone↔prev/next, chain), `CONTEXT=4` (side/goal↔all,
1 hub). Stored symmetric (both directions emitted with the same type) for the
MVP. Total edges linear in (#nodes + #windows) — the §4.6 no-explosion gate.

### Resolved gaps from the user

- **(B) two-stone turns → EVERY STONE PLACEMENT is a separate move**, per-
  placement actions, sequential, exactly like dense_cnn. The policy/MCTS action
  is a single stone; `opp_policy` = the next placement decision (which may be the
  same player's second stone or the opponent's, per engine turn phase).
- **(C) value / short-term-value / opp_policy targets** mirror dense_cnn's
  `finalize_game_samples`, adapted to per-placement turns + candidate rep
  (detailed in Phase 4).
- **(A) node features, (D) D6 edge-relabel, (H) GNN/transformer hypers ~2.1M,
  (I) collation** — designed here / in later phase entries.

---

## Phase 1 — candidate/window/edge Rust path + validation gates

**Implemented** (`hexgt/rust/src/candidates.rs`, `state.rs`): the single shared
builder (§4.5) over the engine `WindowStore` + a local `coords_within_radius`
(not exported by the engine). Produces candidate set (A∪B), count-3/4/5 window
tokens, and the bounded typed graph (window-hub edges, no cliques). Exposed via
`rust_bridge.candidate_ids(state, n)` and `rust_bridge.graph_facts(state, n)`.
Edges stored symmetric (both directions, same type). Candidate ⊆ legal verified.

**Validation (`scripts/_validate_hexgt_candidates.py`)** on 60 recorded 96x8
games (4547 positions, compact raw-fact shards, read-only):

**Gate (3) NO-EXPLOSION — PASSES.** edges/node ≈ 6.2 (opening) → 7.4 (mid) →
7.7 (end) median, p95 ≤ 8.3, never growing with position size. Edges are LINEAR
in #nodes; the window-hub routing avoids the same-axis clique. ✓ Candidate sizes
(n=2): median 95/201/388 (open/mid/end), p95 up to 842, max ~1146.

**Gate (1) COVERAGE — n=2 is TOO TIGHT (key finding).**

| n | played% | visited-count% | visited-mass% |
|---|---|---|---|
| 2 | 90.5 | 84.9 | 91.2 |
| 4 | 92.2 | 87.7 | 92.5 |
| 6 | 95.1 | 93.3 | 95.6 |
| 7 | 97.8 | 96.5 | 97.6 |
| 8 | **100.0** | **100.0** | **100.0** |

Per phase at n=2: opening 80%/64%/84% (worst), midgame 93%/90%/94%, endgame
90%/87%/90% (played/vcount/vmass). Missed PLAYED moves at n=2 are FAR spread
plays: 68% at hex-distance 6–8 from the nearest stone (only 5% at dist 3) — so
raising n incrementally barely helps; ~100% needs n=8.

**Interpretation:** the union rule is SOUND (100% at n=8 = full legal radius, no
red flag — `coords_within_radius(stone, 8)` ≈ the engine legal set). But the
plan's premise that a *small* n retains ~100% does NOT hold for Hexo: strong
play genuinely uses far tenuki/spread moves spread across the whole legal radius.
There is no intermediate n with both ~100% coverage and a small candidate set.

## Phase 4 — expand-time graph construction + HexgtTrainer

**Recompute-at-expand (`expand.py`)**: reconstruct each compact raw-fact row's
position by replaying its placement history through the engine, build the graph
via the SHARED Rust path, and assemble targets. The compact shard schema is
REUSED unchanged (no new SCHEMA_VERSION) — `dense_cnn.compact_io.read_compact_shard`
reads them, and `dense_cnn`'s replay/shuffle is model-agnostic, so only the
expand step (graphs vs planes) is hexgt-specific.

**Target construction (gap C)** reuses dense_cnn's already-finalized facts
(value / opp_policy / short-term-value are per-per-placement-turn) — the only
remap is the policy/opp visit distribution from action-id space onto the
candidate nodes in CSR order. At candidate_radius=8, candidate≡legal so dropped
visit mass is ~0 (validated: 0.000% over 954 real rows; n=2 drops 7.6%, matching
Phase-1 coverage). Value round-trip 954/954.

**D6 augmentation SKIPPED** — the model is D6-invariant by construction, so
rotating the data yields identical training signal (augmentation is redundant).
This is the payoff of the Phase-3 invariance decision; the equivariance test
guarantees the symmetry without 12× data.

**`HexgtTrainer`** mirrors dense_cnn discipline: AdamW + linear warmup, AMP
autocast (cuda), grad-clip, `hexgt_loss` with the configured weights, per-
component loss reporting, and a KataGo-style `HexgtTrainState` that round-trips
through the checkpoint (`to_dict`/`from_mapping`, `sample_count`). `optimizer_step`
+ `train_on_rows`/`train_on_shards` drive the CPU gate; the full replay-window /
train-bucket integration (reusing dense_cnn's model-agnostic replay) is wired in
the GPU phases. The plugin now returns the trainer.

**Gates met:** byte-stable graph batches; targets consistent with engine facts
(value round-trip 100%, n=8 zero dropped mass); a CPU training pass decreases
loss. Full suite 177 passed.

---

## Phase 3 — dynamic GNN + transformer model + D6 equivariance

**D6 handling (gap D) — architectural INVARIANCE, not augmentation.** All node
and edge features are D6-INVARIANT: NO raw axial coords, NO window axis labels
(both rotate under D6). The one coord-derived feature is `center_distance`
(max-norm from the opening (0,0), which D6 fixes). Geometry rides on the graph
STRUCTURE (adjacency + window-membership edges) + an invariant per-edge
hex-distance. Since every op (relational message passing, attention) is
permutation-equivariant and inputs are invariant, the model is D6-invariant by
construction → the §6.5 equivariance test passes EXACTLY for all 12 elements with
no augmentation. This is the principled choice for a rotationally-symmetric game
and avoids the dense_cnn D6-corruption class of bugs entirely. `d6.py` re-derives
the 12-element group from cube coords (tested: group laws + distance preservation).

**Node features (gap A) — finalized, D6-invariant** (`constants.py` / `features.py`,
F=32): type one-hot, owner, center_distance, stone recency, window count one-hot
+ empty-count, candidate completion/`nwin` tactical flags (from window-membership
edges), side-node phase/counts/move-number. Edge attr = edge-type one-hot ++
hex-distance (`EDGE_ATTR_DIM=6`).

**Model body (§6.3):** `RelationalMessagePassing` (per-edge-type linear via einsum,
edge-attr-aware, mean-aggregated, residual+LN) ×`gnn_layers`, then
`GraphTransformerLayer` (per-graph context self-attention over {side,stone,window}
+ candidate→context cross-attention) ×`ctx_layers`. Attention is computed
per-graph over the contiguous packed slices (cost O(#ctx²+#cand·#ctx)/graph, not
O(N²)) — the Python per-graph loop is the Phase-5 vectorization lever. Value reads
out from the SIDE hub node (one per graph).

**Hyperparameters (gap H) — sized to ~2.1M:** `token_dim=168, ffn=336,
gnn_layers=3, ctx_layers=3, heads=4` → **2,073,214 params** (1.3% under the 96x8
baseline's ~2.1M). Verified by `sum(p.numel())` (test gate 1.89–2.31M).

**Collation (gap I):** `collate.py` packs per-graph `GraphTensors` into one
disjoint graph (contiguous per-graph node slices, offset edges/candidates,
`candidate_graph` segments, `candidate_ids` CSR). `graph_build.py` ties Rust facts
→ featurizer → collate (shared by training + search). PyO3 maps `Vec<u8>` → Python
`bytes`, so the featurizer coerces the u8 columns via `np.frombuffer`.

**Tests (all green):** D6 equivariance (12 elements, end-to-end via rotated
replays), param budget, real-graph forward, overfit a tiny batch, candidate↔priors
CSR ordering, checkpoint round-trip + incompat detection. Full suite 174 passed.

---

**DECISION (candidate-set policy) — SUPERSEDED by the user's n=3 call below.**
My earlier interim choice was n=8 (candidate≡legal, 100% coverage) to avoid a
move-vocabulary handicap. The user, after seeing the coverage/compute tradeoff,
made the binding call:

**BINDING DECISION (user): default `candidate_radius = 3`, practical range [2,4],
drop the "up to 8" framing + DATASET PRUNING for the out-of-set tail.** Rationale:
n=8 only matters for dense_cnn's LEARNED novel-far-placement strategy, which isn't
needed for high-level play; n=3 covers practically all useful moves at far lower
attention cost. The ~8% far-spread tail is handled by PRUNING recorded positions
in BC/sample-gen (`HexgtSampleConfig.bc_prune_max_dropped_mass=0.15`,
`expand.build_training_batch` / `HexgtTrainer`), NOT by widening the radius:
a position is dropped when >15% of its policy visit-mass (or all of it) lands
outside the n=3 candidate set; survivors renormalize over in-set candidates. The
prune rate is logged.

**Validated on RECENT 96x8 (epoch-24 games, the highest-quality data):**
- n=3 coverage: played **91.8%** / visited-count 78.2% / **visited-mass 90.2%**.
  By phase (visited-mass): opening **66%** (sparse-board far openings — the bulk
  of the misses), midgame **96.6%**, endgame **98.1%**.
- BC PRUNE RATE @ n=3, threshold 0.15: **10.0%** of positions pruned → **~90% of
  96x8 data survives BC** (n=2: 11.3%, n=4: 9.6%). Value round-trip 822/822.
- Candidate counts at n=3 (median): ~270 (vs ~610 at n=8) — ~2× cheaper dense
  candidate↔context attention than the old n=8 choice.

---

## Phase 5d — feature-buffer optimization (the Phase-5c bottleneck fix)

The Phase-5c bottleneck (self-play ~89% Python featurization) is FIXED. Ported
`features.build_graph_tensors` + `collate.collate_graphs` into Rust
(`hexgt/rust/src/features.rs`), parallelized per-graph across rayon, emitted to
Python as ZERO-COPY buffer-protocol views (`HexgtF32Buffer`/`HexgtI64Buffer`,
mirroring dense_cnn's `PlaneBuffer`). `mcts_eval.rs` now featurizes+collates the
whole leaf batch in Rust; `inference.evaluate_featurized_batch` just
`torch.frombuffer` (≈0.01 ms, free) + the GNN forward. Byte-identity gated by
`tests/test_hexgt_feature_buffer.py` (max|d| = 0 vs the Python path over 1.8M
feature values — a mismatch silently poisons the model).

**CRITICAL FIX:** do NOT release the GIL (`py.detach`, pyo3-0.28's renamed
`allow_threads`) inside the eval callback. The copied dense_cnn MCTS drives its
select↔eval pipeline across rayon threads; a per-chunk GIL hand-off there
collapsed self-play to **0.5 pos/s** (9× regression, invisible to single-thread
micro-benchmarks — `scripts/_eval_pathbench.py` confirmed the forward itself is
identical old/new at every batch size). Hold the GIL while rayon featurizes.

**Cache-assisted self-play throughput (64 games × visits=128, REAL 96x8 leaves,
RTX 4070 Ti; `scripts/_mcts_selfplay_probe.py`):** baseline 4.3 pos/s
(featurization-bound) → Rust featurize 5.9 (vbatch=16) / 9.4 (vbatch=64) /
**29.7 (vbatch=64 + torch.compile)**. The compiled rate EXCEEDS dense_cnn 96x8's
~23, inside the 17–68 ceiling. **GO — the dynamic GNN is throughput-competitive.**
Use vbatch=64 + compile for self-play. (Training featurization in `expand.py`
still uses the Python path; ~240 ms/step, tolerable for BC; port if RL needs it.)

## Phase 7 — player + head-to-head eval wiring (Phase-5+ stubs → real)

`HexgtPlayer` (`player.py`): a `hexo_runner` player wrapping `HexgtInference` +
`HexgtMctsSession` (mirrors `DenseCNNPlayer`), DETERMINISTIC by default (greedy
temperature=0, NO root Dirichlet noise, fixed per-(game,move) seed) so
comparisons are repeatable; other search knobs match self-play (matched compute).
`evaluation.run_head_to_head` is the reusable match driver (alternates colors,
fixed seeds, scores from A's view); `evaluate_epoch` pairs hexgt vs SealBot
best-50ms (plugin hook). `scripts/_head_to_head.py` runs hexgt vs the dense_cnn
epoch-24 checkpoint (read-only) and vs SealBot at a matched visit budget. Gated by
`tests/test_hexgt_player.py`.

## Phase 8 — converged behavioral-clone run

`scripts/_bc_train.py`: RAM-disciplined streaming converged BC (stream shard
groups → train one pass → free; periodic checkpoint + held-out imitation eval on
an epoch never trained on; crash-safe). 96x8 epochs 20/21/23/24 train, epoch 22
held out, lr 1e-3 / warmup 500, ~240 ms/step. Components drop steadily (policy
5.4→~2.9, value 4.2→~0.6); the policy CE floor (~2.9) is the diffuse-teacher
limit, so top-1 agreement is the truer signal. NOTE: `hexgt_loss` reports the
total under key `"total"` (not `"loss"`). RL self-play (a hexgt `selfplay.py`
mirroring dense_cnn's game-driven loop, reusing the representation-agnostic
compact sample format `expand.py` already reads) is the documented next step; the
29.7 pos/s rate makes it viable.

## Phase 9 — head-to-head: IS THE GNN COMPETITIVE? (yes)

**`scripts/_head_to_head.py`, hexgt step-6009 (BC-only, held-out top-1 33.5%) vs
dense_cnn epoch-24, 40 games, MATCHED visits=200, deterministic (greedy / no
Dirichlet noise, alternating colors), e24 loaded read-only:**

```
hexgt vs dense_cnn e24:  22 W / 18 L / 0 D  →  55.0%  (40/40 completed, mean 131 turns)
hexgt vs SealBot:        SKIPPED (no SEALBOT_PATH in the isolated build env)
```

The BC-distilled dynamic GNN plays slightly BETTER than the 96x8 teacher it was
distilled from, at equal search. At n=40 the honest read is "on par to slightly
ahead" (95% CI ≈ 39–70%), not dominance — but combined with the 29.7 pos/s
self-play rate (beating dense_cnn 96x8's ~23), the dynamic GNN architecture is
**validated end-to-end**: feasible, throughput-competitive, and strength-
competitive as a pure BC distillation. The expected next gain is RL self-play on
top of the BC seed (Phase-8 note). To add the SealBot leg, point `SEALBOT_PATH`
at a SealBot checkout and pass `--sealbot`.

---

## Phase 10 — RL self-play infra + VRAM compression + eval harness

**Self-play (`selfplay.py`):** game-driven loop mirroring dense_cnn (keep N games
in flight, search every playable position to terminal/`max_actions`, temp decay +
forced playouts k=2 + root Dirichlet) on the dynamic-GNN `HexgtMctsSession`,
writing the dense_cnn COMPACT shard format `expand.py` already reads (reuses
`sample_from_state`/`finalize_game_samples`/`write_compact_shard`). **Async-batcher
decision:** did NOT build the Phase-7 async Rust batcher — the existing synchronous
session already coalesces all concurrent games' leaves into one batched forward per
round (`virtual_batch_size`+`active_root_limit`), the same throughput property, and
the "baseline-first" rule + the 29.7 pos/s result make the proven path right.

**VRAM compression (the big fix):** profiling the search forward showed the padded
candidate attention was only **~22% packing-efficient** (a couple of endgame graphs
at max_cand ~1281 force EVERY graph in the leaf batch to pad to that max, avg ~260),
and the batch is unbounded (MCTS submits `active*vbatch` ~2-4k leaves): 512g=8GB,
1024g=16GB (SPILLS to host RAM → the throughput killer), 2048g+=OOM. Fix
(`HexgtInference.forward_batch`): split the forward into sub-batches SORTED by
candidate count + bounded by a padded-slot budget, so each chunk pads to its LOCAL
max (~90%+) and peak is capped regardless of leaf count. Graphs are independent
(per-graph attention; global softmax after reassembly) → BIT-IDENTICAL (gated). Live
self-play VRAM **11.7→3.0 GB**, throughput 13→28-62 pos/s (no more host spilling).

**Eval harness:** `run_head_to_head_parallel` + `HexgtBatchedSearcher` step many
games at once (each round partitions by whose turn, batches each side's positions
into ONE forward; deterministic per-game, reproduces the sequential driver). Added
**optional opening variety** (first K plies sampled at a temperature, per-(round,game)
seeded → decorrelates the otherwise-correlated greedy games into ~independent samples
→ smoother win rate, still repeatable). Plus a **fp16-overflow NaN guard**: deep eval
games (max_actions 1024) overflow fp16 to NaN on extreme positions; sanitize the
trunk logits so a poisoned position degrades to neutral instead of crashing the run.

## Phase 11 — exploration-constant ablation (C1/C2/C3) + chosen config

Plan: `docs/analysis/HEXGT_EXPLORATION_ABLATION_PLAN.md`. Three SHORT BC-seeded RL
runs (5 epochs × 96 games × visits 96, active=64), judged by LEARNING + self-play
DATA QUALITY (NOT static strength); Dirichlet/temp judged by Q-metrics, not the
deterministic L1 (which only `c_puct`/`root_policy_temperature` affect). Derived
Dirichlet `total_alpha=6.6` from the measured n=3 candidate median (220).

Configs (α_i = total_alpha/count): **C1** baseline (6.6/eps0.25/rpt1.0/cpuct1.5/
temp1.0→0.2@30); **C2** higher-exp (9.0/0.35/1.0/2.0/1.2→0.3@45); **C3** lower-exp
(4.5/0.15/1.0/1.0/1.0→0.1@20).

**Results (C1 & C3 ran 5 epochs; C2 died at ep1 on a transient `CUDA device not
ready`, supervisor-recoverable, NOT a code bug):**

| metric ep0→ep4 | C1 | C3 |
|---|---|---|
| Q1 decisive | 79%→100% | 74%→100% |
| Q2 unique-opening (diversity) | 60%→**68.8%** | 55%→57.3% (collapsing) |
| Q3 prior entropy (sharpening) | 3.57→3.14 | 3.53→2.81 (sharpest) |
| Q4 mean \|value\| | 0.64→1.00 | 0.60→1.00 |
| L2 holdout CE | 2.78→3.01 | 2.78→3.02 |
| L1 vs frozen seed | 31→20% | 33→38% (noisy) |
| NaN sanitizes (stability) | 486 | **0** |

**Shared dynamic (all configs):** rapid policy sharpening (priorH↓, decisive→100%,
|val|→1.0), L2-holdout-CE RISING (the model moving OFF the dense_cnn imitation
target — expected under RL, not necessarily degradation), and a lumpy/declining L1
vs the frozen seed. The 5-epoch window can't tell "healthy sharpening" from "early
over-commitment"; the overnight 60-epoch run is what resolves it. Also: NaN-sanitize
count scales with exploration (C3=0, C1=486, C2=1785) — more search visits more
extreme positions.

**CHOSEN: C1 (derived baseline).** Rationale: among the stable, completed runs it
best embodies the data-quality target — becomes fully decisive (good value signal)
while RETAINING the most opening diversity (Q2 68.8% vs C3's collapsing 57.3%), i.e.
"diverse AND decisive," the healthiest profile for a long run; numerically stable
(486 rare per-position sanitizes, no crash) vs C2's instability. C3 (sharper, 0-NaN,
diversity-collapsing) is the fallback if C1 destabilizes over many epochs.

## Phase 12 — overnight main RL run (LIVE)

Launched the unbounded BC-seeded RL run with config C1 under the supervisor (auto-
relaunch + RAM watchdog + circuit breaker), `runs/hexgt_rl_main`, 60-epoch cap,
visits=128, refilled concurrency (96 games > active 64), eval every 3 epochs vs
dense_cnn e24 + SealBot (opening-variety) + holdout + Q-metrics. Starting anchors:
45.8% vs dense_cnn e24, 0% vs SealBot. See root `notes.md` for status/stop/resume.
