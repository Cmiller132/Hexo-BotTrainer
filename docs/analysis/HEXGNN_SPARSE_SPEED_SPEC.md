# hexgnn sparse-graph speed spec — efficient re-representation, not deletion

**Goal:** make hexgnn's per-position graph carry the **same information the model
sees today, at lower cost** — so the GNN forward (memory-bound on edge-scale
scatter/gather, ~60% of self-play wall) and the Rust featurizer (O(edges), ~38%)
both get cheaper **without moving the model's behavior**. Throughput target: kill
the 0%-idle GPU gaps and push pos/s up; we measure **how far the model's view
actually moved** (KL on policy, Δ on value) at every gate, not just tactical
safety.

This is a **moderate rewrite of the hexgnn lineage only** (forked Rust crate
`packages/hexgnn/rust/` -> `hexo_models._rust.hexgnn`; the live hexgt module is
untouched). It is **not** a new model and **not** a behavior simplification.

---

## 0. Non-negotiable: TSS is fully kept (and is separable from the GNN graph)

Verified in the code: `threats.rs` operates on the engine **WindowStore**
("no graph/feature construction, no network", `threats.rs:3`). `tactical_cells()`
and `min_hitting_set()` read `state.board().windows()` directly. The entire TSS
stack — WindowStore threat index, tactical injection at expansion (union of
opponent >=4-window empties + own win-now cells, additive cap, forced first
visits; `mcts_tree.rs:412-425,509-514,815-946`), phase-aware hitting-set leaf
overrides (HARD WIN/LOSS), and the tactical move-selection guard — reads the
WindowStore, **NOT** the sparsified GNN graph. So sparsifying the GNN graph
**cannot** blind TSS.

The ONE coupling: TSS injects tactical cells as tree children whose **priors come
from the GNN eval**. Behavior is preserved iff every tactical cell remains in the
candidate set. **Invariant T0:** the sparse candidate set force-includes
`tactical_cells(state)` at radius 0 in every phase (it is already a strict subset
today). Gate: the **full TSS test suite** (injection-additive, hitting-set
override, move-guard, two-stone defense, VCF) stays green at **every** phase.

---

## 1. Where the cost is (measured, td128/gnn3, active=512, visits=512)

- Wall: **GNN forward (GPU) ~60%**, Rust MCTS+featurize ~38%, py glue ~1.5%.
- Forward op mix: matmul only ~13-26%; the rest is **elementwise + scatter/gather
  over the edge-scale tensors** (`index_add_`, `index`, `add`, `clamp_min`,
  `copy_`) => **memory-bandwidth-bound on EDGES**, partly launch-bound in eager
  (compile recovers 2.2-3x).
- Per-leaf graph grows ~7x opening->ply60: nodes 222->1569, **edges 1499->11420**,
  candidates 215->1507. Edges are the enemy.

Edge classes today (`candidates.rs:312-377`), directed counts:
| class | construction | count driver |
|---|---|---|
| **CONTEXT** | SIDE hub <-> EVERY other node | `2*(N-1)` — grows with N |
| **ADJACENCY** | stones+candidates within hex-dist 1 | ~6 * spatial nodes |
| CANDIDATE_WINDOW | window <-> its empty cells | <=6 per window, **overlaps** |
| STONE_WINDOW | window <-> its one-color stones | <=6 per window |
| RECENCY | consecutive stones | ~2 * stones |

CONTEXT and ADJACENCY dominate; both scale with **candidate count**.

---

## 2. The sparse representation — each edge class, re-represented to keep its contribution

Principle: for each edge class, identify *what computation it feeds the model*,
then deliver that computation at lower cost. Three tiers, by behavioral risk.

### 2A. CONTEXT hub -> analytic broadcast (BEHAVIOR-EXACT, the flagship win)
**What it contributes:** every node reads a 1-hop message from the SIDE hub, and
SIDE reads a mean message from all nodes. Message into node `i` =
`relu(W_ctx · h_side + edge_proj(attr_{side->i}))`, where `W_ctx · h_side` is
**identical for all i** and only `edge_proj(attr_i)` (type one-hot + hex-distance)
varies. SIDE's incoming = `mean_i relu(W_ctx · h_i + edge_proj(attr_{i->side}))`.

**Re-representation:** compute the CONTEXT class **analytically, with zero
materialized edges** — precompute the shared `W_ctx·h_side` once, add the per-node
`edge_proj(attr)` as a dense `(N, d)` op, `relu`, and fold into each node's
aggregation; compute SIDE's incoming as a segment-mean over nodes. This is the
**exact same output** (fp-identical up to reduction order), but removes the single
largest edge class (`2*(N-1)`) from the scatter/gather. Touches the message-
passing kernel (`architecture.py` RelationalMessagePassing) + the graph builder
(stop emitting CONTEXT edges; flag the class as analytic). **Closeness: exact
(KL~=0, |Δv|~=1e-6).** Biggest single forward win, zero behavior change.

### 2B. Window-hub edges -> dedup + shared aggregation (NEAR-EXACT)
**What it contributes:** a candidate/stone connected to its windows receives one
message per incident window; overlapping windows (same cell in multiple lines)
emit **duplicate CANDIDATE_WINDOW edges** to the same candidate. The GNN
mean-aggregates them, so what the candidate actually sees is the **mean window
message**, and the per-window count/owner is **already in the candidate's features**
(`F_CAND_OWN/OPP_WIN{3,4,5}`, `F_CAND_NWIN_*`).

**Re-representation options (closeness-gated, pick by metric):**
- (i) **Dedup**: collapse parallel candidate<->window edges of the same
  (owner,count) into one weighted edge (weight = multiplicity) so the mean is
  preserved exactly under a multiplicity-weighted aggregation. Near-exact.
- (ii) **Aggregated window-class hubs**: replace per-window tokens with up to 6
  aggregated hub nodes per (owner in {own,opp} x count in {3,4,5}); a candidate
  connects to the hub(s) it belongs to. Cuts window nodes+edges hard midgame.
  Changes the message (per-window geometry is lost, but it is D6-non-invariant
  anyway and largely captured by features) -> **must pass the closeness gate**.
Default to (i) (near-exact); promote (ii) only if the closeness metric stays
within gate AND the TSS suite stays green (TSS does not read these edges, so it
will).

### 2C. Candidate set + adjacency -> relevance-capped, cold-shed (SMALL, MEASURED move)
**What it contributes:** ADJACENCY gives local spatial structure; the radius-n
filler gives the model "nearby empty" options. Today the set is
`active-windows ∪ n-radius(open-line)` with a dead-cell prune already in place
(`has_open_window`, `candidates.rs:77-98,168-184`).

**Re-representation:** keep **every tactically-live cell** (all active-window
empties + all `tactical_cells`), shed only **provably-cold** cells:
- **Phase-gated radius**: n=2 while `move_number < R_LATE_AFTER_MOVE` (default 30),
  else n=3. Early boards are sparse, so radius-2 already covers the live region;
  the documented coverage gap n2-vs-n3 is ~1% of strong-move mass.
- **Relevance cap, not blunt radius**: a radius cell is kept iff it is in an active
  window OR has an open completable line (existing `has_open_window`) — extend to
  also keep cells adjacent to a stone (preserves ADJACENCY locality). Cells that
  are cold by ALL criteria are dropped.
- **Adjacency**: keep every edge with a stone endpoint or a tactical endpoint; drop
  only candidate<->candidate edges in deep empty regions (both endpoints cold) —
  these feed the model almost nothing (uniform empty neighborhood). Closeness-gated.

All thresholds are **config knobs with current behavior reachable** (n=3 always,
context analytic-off, no dedup, adjacency unrestricted) so every change is
A/B-testable and revertible.

---

## 3. Behavioral-closeness metric (new hard gate)

For a corpus of ~512 real positions spanning ply bands (opening/mid/late), run the
**same trained (or fixed-seed random) weights** through CURRENT vs SPARSE
representation and compare the heads:
- **policy**: per-position `KL(softmax(policy_current) || softmax(policy_sparse))`
  over the shared candidate set; report mean + p95.
- **value**: `|E[value_current] - E[value_sparse]|` (decoded scalar); mean + p95.
- **opp_policy**: same KL as policy.

**Gates** (per tier): 2A exact (mean KL < 1e-4, |Δv| < 1e-3). 2B/2C near: **mean
KL < 0.03, p95 KL < 0.10, mean |Δv| < 0.02, p95 |Δv| < 0.05** (tightened if the
owner wants closer). Any sparsification exceeding the gate is reverted or demoted
to the exact tier. This quantifies "how far the model's view moved" — the owner's
requirement — independent of tactical safety.

---

## 4. Expected edge reduction per ply band (estimate; re-measured at the gate)

| ply band | lever(s) | edges/leaf now | edges/leaf sparse | cut |
|---|---|--:|--:|--:|
| opening (<10) | 2A + n2 | ~1,500 | ~600-700 | ~55% |
| early-mid (10-30) | 2A + n2 + dedup | ~4,000 | ~1,900-2,300 | ~45% |
| midgame (30-60) | 2A + adjacency-cold + dedup | ~11,400 | ~5,500-6,500 | ~45% |

2A (CONTEXT analytic) removes `~2*N` edges at every ply (largest at midgame where
N~1569 -> ~3100 directed edges gone, exactly). Candidate/adjacency cuts add on top
early-mid. **Direct measurement already in hand:** n=3 -> n=2 cut edges/leaf
1972 -> 1370 (-30%) and raised pos/s 28.0 -> 41.2 (+47%) on td96/gnn2 @512.

---

## 5. Rust optimization (the 38%)

1. **Single live-cell pass**: replace the per-candidate `has_open_window` O(18)
   rescan with one forward sweep building a `HashSet<cell>` of live cells, then
   membership-test radius cells — O(windows*6 + candidates) vs O(candidates*18).
2. **Static-topology reuse**: within one move's search the leaves differ from the
   root by few stones; cache the live-cell set + window enumeration keyed by board
   occupancy (not full state hash) and reuse across that position's leaf builds.
3. **Dedicated featurize threadpool** so `build_graph`/featurize don't contend with
   `select_leaf_batch`'s rayon over 512 roots; raise rayon grain (batch tiny
   opening graphs per task to cut spawn overhead).
4. Keep the zero-copy buffer output + pinned scatter unchanged (parity-preserving).

## 6. Pipeline (kill the 0% GPU gaps)

The session already has a select<->eval prefetch (`mcts.rs:440-545`) and a
featurize<->forward double-buffer (`mcts_eval.rs:146-216`).
1. **Deepen** `HEXGNN_EVAL_PIPELINE_DEPTH` 2 -> 3/4 so the featurizer queues ahead
   while the GPU runs (now that the forward is ~60%).
2. **Verify/force GIL release** around `_host_to_device` + the forward
   (`inference.py:224-260`) so the GIL-free Rust featurizer actually overlaps the
   CUDA launch + H2D; drop the GIL if not.
3. **Pinned-host ring buffer** (2-3x largest chunk) instead of per-call
   `pin_memory()`.
4. Optional later: CUDA graphs on bucketed shapes + a fused relational-message
   scatter kernel (the `index_add_` is ~13-16% of forward CUDA time).

---

## 7. Phased build + gates (each phase: commit via clone, report numbers)

- **WS0 — fork + rebuild (unchanged):** register `hexo_models._rust.hexgnn`, point
  hexgnn `rust_bridge` at it, maturin rebuild. **Gate:** featurizer parity (Rust
  hexgnn vs Python), D6 equivariance, full TSS suite — all green on the
  byte-identical fork. (Proves the fork/rebuild before any behavior change.)
- **WS1a — CONTEXT analytic (2A):** **Gate:** closeness EXACT (KL<1e-4), parity,
  D6, TSS green; edges/pos -2N; pos/s up.
- **WS1b — window dedup + relevance candidate set (2B/2C):** **Gate:** closeness
  within near-gate, parity (or updated featurizer parity if both halves change),
  D6, **full TSS suite**, edges/pos per table; pos/s up.
- **WS2 — Rust opt (sec 5):** **Gate:** featurize ms down, parity green, pos/s up.
- **WS3 — pipeline (sec 6):** **Gate:** GPU util saturated (no 0% gaps), pos/s up.
- **FINAL gate:** **>=100 pos/s @512 visits no-PCR on FULL-GAME self-play** (or the
  honest best with the config that hits the owner's throughput goal), all quality
  gates green, then HOLD for the launch decision.

Quality gates at EVERY phase: featurizer parity (<1e-6), D6 equivariance (all 12
elements), the **full TSS test suite**, the **closeness metric**, and shard sanity
(lambda=0 hard targets, recorded rows, opp-mask).

---

## 8. Honest ceiling note

Triangulated (profiling + design + the n=2 measurement): the "don't-go-too-far"
sparse rewrite realistically reaches **~40-65 pos/s opening / ~30-45 full-game at
512 visits (~2x)** with the GPU saturated. The CONTEXT-analytic win (2A) is the
largest single behavior-exact lever. Hitting **>=100 @512 full-game** likely also
needs a lower average visit count (e.g. visits ~128-192) OR promoting the
behavior-near tier (2B option ii) — both surfaced to the owner as decisions, gated
by the closeness metric so we always know how far the model moved.
