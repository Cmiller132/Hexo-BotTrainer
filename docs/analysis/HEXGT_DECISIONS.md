# HEXGT (Model 2) — Implementation Decisions Log

Running log of non-trivial design decisions made while building `hexgt` (the
dynamic GNN + transformer Model 2) per `HEXFORMER_REWRITE_PLAN.md`. Each entry
records the decision, the plan reference, and the rationale. The plan's §11
"Open questions" and §12 readiness-gaps (A–J) are resolved here as implemented.

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
