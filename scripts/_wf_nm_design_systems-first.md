# hexfield — a stone-anchored, native-hex lineage designed backward from the codebase

Design competition entry — lens: CODE ARCHITECTURE AND INTEGRATION.
All file:line citations verified against the working tree on 2026-06-12.

---

## 1. Identity

**hexfield** is a crop-free, native-hexagonal policy/value network whose domain is the
stone-anchored support set (`stones ∪ full legal set ∪ 1-ring halo`), evaluated by the repo's
*proven* batched-PUCT machinery rather than a re-derived one. The design's central systems move is
to split the work into exactly three artifacts with sharp contracts: (1) a new shared crate
`hexo_search` created by **extracting** the already-model-agnostic search core (tree + continuous
scheduler + eval cache + state intake) out of dense_cnn — the same playbook that produced
`threats_shared` — gated by byte-determinism golden tests so dense_cnn's behavior is provably
unchanged; (2) a greenfield `packages/hexfield` lineage (own Python package, own Rust crate, own
cdylib `hexfield._rust`) containing only what is genuinely new: support-set featurizer, node-set
network, evaluator payload, data pipeline; (3) a parity-oracle test suite in which the trusted
lineage's code is imported *as executable specification* (never forked) — the Rust featurizer is
pinned to the Python expander, the 7-tap node conv is pinned to dense_cnn's masked 3×3 on embedded
patches, and the sdpa attention is pinned to a materialized oracle. Every owner-locked architecture
decision (§2 of the brief) is honored; every search semantic (PUCT, widening, FPU, virtual loss,
Dirichlet, tree reuse, transposition cache, PCR, policy-init, mix_seed streams, TSS toggle) is
preserved *by construction* because it is the same single-sourced code, not a reimplementation.
The build story keeps the live main_4 run untouchable: building hexfield never rewrites
`hexo_models._rust`, and the extraction is inert in the live venv until an owner-scheduled rebuild.

---

## 2. Input representation

### 2.1 Support set

For an engine state `s`:

- `stones(s)` — occupied cells (`state.board().occupied_cells()`).
- `legal(s)` — the engine legal set: empty cells within hex distance 8 of any stone
  (`LEGAL_RADIUS = 8`); at ply 0 (phase `Opening`) the engine forces `{(0,0)}`.
- `halo(s)` — cells adjacent (hex distance 1) to `stones ∪ legal` that are in neither set.
  Geometrically this is the empty ring at hex distance exactly 9 from the stone set (neighbors of
  rim legal cells). Halo cells carry features and participate in conv/attention but **never carry
  logits**.

`support(s) = stones ∪ legal ∪ halo`, in **canonical order: ascending `(q, r)` lexicographic**.
Canonical order is a hard contract (parity tests and the positional prior protocol depend on it).

Invariants (each is a test):

- **I1 (zero coverage loss):** `|legal nodes| == state.legal_move_count()` always.
- **I2 (complete neighborhoods):** every legal node and every stone node has all 6 hex neighbors
  inside `support` (legal cells at distance ≤ 8 have neighbors at distance ≤ 9 ⊆ legal ∪ stones ∪
  halo). Only halo nodes may have missing neighbors; missing neighbors contribute zeros (conv
  zero-padding semantics, brief §3).
- **I3 (ply-0):** empty board ⇒ support = `{(0,0)}` ∪ its 6 neighbors = 7 nodes, 1 legal node.
  No special-case code; the general construction yields this.

Sizes: after ply 0, 1 stone ⇒ 216 legal + 1 stone + 54 halo (ring at 9 has 6·9 cells) = **271
nodes** (the brief's "217" is the radius-8 disk count `1 + 3·8·9` including the stone cell).
Mid-game ≈ 600–1500; tail ≈ 3k (brief §4 measured numbers govern perf planning).

### 2.2 Node features — F = 13 channels, exact formulas

Port of the 13 trusted dense_cnn planes (semantics from
`packages/hexo_models/dense_cnn/rust/src/encoding.rs:105-298` and plane order from
`packages/dense_cnn_restnet/python/dense_cnn_restnet/constants.py:29-41` — order kept identical,
including the opp-hot-before-own-hot quirk, to minimize porting mistakes):

| ch | name | formula (current player = `me`, `P = placements_made()`) |
|----|------|-----------------------------------------------------------|
| 0 | own_stone | 1 if cell stone owned by `me` |
| 1 | opp_stone | 1 if cell stone owned by opponent |
| 2 | empty | 1 − ch0 − ch1 (legal and halo cells are 1) |
| 3 | legal | 1 if cell ∈ engine legal set |
| 4 | second_placement | constant over all nodes: 1 iff phase == SecondStone |
| 5 | first_stone | 1 at the turn's first placement cell, only when phase == SecondStone |
| 6 | player_colour | constant: 1 iff `me == Player0` |
| 7 | own_recency | at own stone with placement index `k`: `1 / (1 + (P − k))` (encoding.rs:182-196 exactly) |
| 8 | opp_recency | same for opponent stones |
| 9 | opp_hot | 1 if cell is an empty cell of an opponent **single-colour count ≥ 4** length-6 window; feature gated `P ≥ 7` (encoding.rs:226-266 exactly) |
| 10 | own_hot | same for own windows |
| 11 | dist_nearest_stone | `min_{t ∈ stones} hexdist(cell, t) / 8`; 0 everywhere when stones = ∅ (ply 0). Stones → 0, legal → (1..8)/8, halo → 9/8 = 1.125 (deliberately > 1: "outside legal range" marker). Replaces crop-center distance. |
| 12 | opp_last_turn | 1 at the 1–2 cells of the opponent's most recent completed turn (encoding.rs:268-298 semantics) |

`hexdist(a,b) = max(|dq|, |dr|, |dq+dr|)`, `dq = a.q−b.q`, `dr = a.r−b.r`.

All features are computed in f32; transport to Python is f16 (loss-free for masks; ≤1e−6 absolute
rounding for ch7/8/11 values; gated by a tolerance test, the dense fp16-gate pattern,
mcts_eval.rs:44-47).

### 2.3 Neighbor table

`DIRS = ((1,0), (0,1), (1,−1), (−1,0), (0,−1), (−1,1))` (engine axes ±Q, ±R, ±QR;
encoding.rs:425-431). For node `i`, `nbr[i][d]` = canonical index of `coord(i) + DIRS[d]` if in
support, else the sentinel `M` (= total node count; consumers append one zero row at index `M`).
Built in Rust with an `ahash` coord→index map (M ≈ 1.5k ⇒ ~9k probes, microseconds); built
identically in Python at expansion time.

---

## 3. Trunk

Channels C = 96, heads H = 4, mlp_ratio 2, interleave **`C C C A C C A C A`** (9 layers, brief §3
default accepted; rationale: receptive radius after stem+3 conv blocks = 1+6 = 7 ≥ window span 5
before any attention; 3 A layers give the token hub ≥ 2 bidirectional rounds; trunk ends on A so
both cells and tokens are fresh at the heads).

### 3.1 HexNodeConv (the direction-typed 7-tap primitive)

For input `x ∈ R^{M×Cin}`:

```
x_pad = concat(x, zeros(1, Cin))                      # sentinel row M
g     = x_pad.index_select(0, nbr.flatten()).view(M, 6*Cin)
y     = concat(x, g) @ W + b                          # W: (7*Cin, Cout), block order [self, D0..D5]
```

One GEMM per conv: mathematically identical *in family* to dense_cnn's `HexConv2d` (3×3 with
corners (0,0)/(2,2) masked = exactly 7 taps, one C×C matrix per relative direction,
architecture.py:163-179). **Equivalence is proven executably** (test T6, §9.4): embed a random
support set into a 41×41 grid, copy the 7 direction matrices into a masked 3×3 kernel, run
dense_cnn's `HexConv2d`, gather at support cells, compare to HexNodeConv ≤ 1e-5.

### 3.2 Conv residual block (×6)

`y = ReLU(x + BN2(Conv2(ReLU(BN1(Conv1(x))))))` — post-activation residual, exactly dense_cnn's
`ResidualBlock` family (architecture.py:237-259). Convs bias-free; **BatchNorm1d over the flat
node dimension** `(M_total, C)` (brief default; flat-concat is padding-free so BN statistics see
only real nodes; train-step BN population ≈ 32 rows × ~900 nodes ≈ 29k samples). `norm = "layer"`
is a config fallback knob (per-node LayerNorm) — kept because BN-over-support is a semantic shift
vs BN-over-fixed-crop (flagged in §12).

Stem: `HexNodeConv(13→96, bias=False) + BN + ReLU` (EmbedNet semantics, architecture.py:730-738).

### 3.3 Attention block (×3) with 8 carried summary tokens

Pre-norm transformer block, restnet family (architecture.py:576-620):
`x = x + Attn(LN(x)); x = x + MLP(LN(x))`, MLP = `Linear(96→192) → GELU → Linear(192→96)`.

**Layout.** Attention runs on per-graph padded tensors. For a group of G graphs padded to bucket
length L (cells): sequence = `[tok_0..tok_7, cell_0..cell_{L−1}]`, shape `(G, 8+L, C)`. The 8
token states are initialized once per forward from a learned `(8, 96)` parameter (broadcast per
graph), enter at the **first** A layer, are **held aside unchanged through conv layers** (the
forward carries `(cells_flat, tokens)` state), and re-enter every subsequent A layer — cells ↔
tokens attend in both directions (owner-locked §2.6). Tokens pass through the block's MLP like any
sequence element. Padded cell rows are garbage-in/garbage-out: their KEY columns are masked
(additive −1e4 in compute dtype), their query rows are discarded at scatter-back, and every real
query row always has ≥ 8 unmasked keys (the tokens), so no all-masked softmax NaN is possible.

**Relative-position bias.** Per A-block learned table `bias_table ∈ R^{229×H}` (no decay; excluded
from AdamW weight decay by the existing name rule, plugin.py:70-78):

- indices 0..216 — exact entries for axial offsets `(dq, dr)` with `hexdist ≤ 8`
  (217 = 1 + 3·8·9), enumerated in ascending `(dq, dr)` order;
- 217..224 — ring buckets for `hexdist` 9..16 (index `217 + hexdist − 9`);
- 225 — far bucket (`hexdist > 16`);
- 226 — cell-query → token-key; 227 — token-query → cell-key; 228 — token → token.

Index construction is **on-GPU from shipped coords**, no O(L²) host traffic:

```
dq = q[:,None] − q[None,:]; dr likewise; dist = max(|dq|,|dr|,|dq+dr|)
idx = where(dist ≤ 8,  EXACT_LUT[(clamp(dq,−8,8)+8)*17 + clamp(dr,−8,8)+8],
      where(dist ≤ 16, 217 + dist − 9, 225))
```

`EXACT_LUT` is a 289-entry (17×17) uint8 buffer built from the documented enumeration (entries with
`hexdist > 8` are never selected). Token rows/columns overwrite with 226/227/228. The additive bias
`bias_table[idx]` is materialized per group in compute dtype (fp16 at inference) and fed to
`F.scaled_dot_product_attention` as `attn_mask`, exactly the restnet pattern
(architecture.py:262-281: sdpa with additive bias ≡ materialized scores; the materialized path is
retained as the test oracle, brief §3). Memory is bounded by the pair-budget batcher (§10.3).

### 3.4 Heads attachment

After the final A block: token slots split **2 (main value) / 2 (STV + moves-left aux) / 4
uncommitted hub capacity**; cell rows scatter back to flat `(M, C)` for the policy heads.

---

## 4. Heads & losses

All targets and masks are built at expansion time (§7.2); loss weights are config, defaults =
production values (brief §4).

| head | architecture | output shape | target | loss | weight |
|------|-------------|--------------|--------|------|--------|
| policy | `HexNodeConv(96→96, bias) → ReLU → Linear(96→1)`, logits gathered at legal nodes | flat `(T,)`, CSR rows | MCTS visit policy: weights at action ids mapped to legal-node positions; row-normalized | segment soft-CE: per-row `log_softmax` over the row's legal segment; **no −1e9 masking exists anywhere** — the logit support *is* the legal set | 1.0 |
| opp_policy | same shape, separate parameters | flat `(T,)` | next decision's visit policy projected onto **this** position's legal set; mass at off-support ids dropped (spill counted, §7.2); zero rows (fast/init next move, or full spill) contribute 0 (`allow_zero_rows` semantics, losses.py:157-158) | segment soft-CE | 0.25 |
| value (main) | `concat(tok0, tok1) (192) → Linear(192→64) → ReLU → Linear(64→65)` — **private pathway, no sharing with aux** (heads_v3 lesson, architecture.py:664-675, applied natively via dedicated tokens) | `(B, 65)` | hard z ∈ {+1,−1} from the mover's perspective → adjacent-bin soft target over the 65-point support `linspace(−1,1,65)` (losses.py:33-53 formulas) | binned soft-CE | 1.0 |
| stvalue_2/6/16 | `concat(tok2, tok3) → Linear(192→64) → ReLU` shared aux body → 3 × `Linear(64→65)` tops | 3 × `(B, 65)` | per-horizon EMA of future root values stepped over FULL turns, horizons in decisions [2, 6, 16]; semantic source `dense_cnn_restnet/samples.py:357+` (ported, oracle-tested) | binned soft-CE, per-head mask | 0.1 |
| moves_left | 4th top on the aux body: `Linear(64→65)` | `(B, 65)` | raw decisions-remaining `ml` stored in shard; at expansion: `2·min(1, ml/512) − 1` → binned (cap = 512, constants.py:16-27); mask `ml ≥ 0` | binned soft-CE, masked | 0.1 |

Value/STV/ML reuse one shared `65-bin` helper module (`value_bins`, `scalar_to_binned_target`,
`decode_binned_value`, strict finiteness/mass checks — reimplemented fresh with losses.py:20-131 as
the semantic reference and a test oracle). Rows from truncated games are **dropped at write time**
(main_4's `drop_truncated_rows = true` becomes the lineage default — z=0 rows are confirmed
poison). Total loss = Σ weights·components; AdamW lr 1e-3, wd 1e-4 on matrix weights only
(excluded: ndim ≤ 1, `bias_table`, the `tokens` init matrix), AMP, grad-clip 1.0.

---

## 5. Symmetry — D6 by training-time augmentation

Owner-locked: augmentation, not architectural invariance. The 12 transforms are exactly
dense_cnn's (`d6.py:79-136`): index ≥ 6 ⇒ reflect first `(q,r) → (q, −q−r)`, then `index mod 6`
rotations `(q,r) → (−r, q+r)`, around center `(0,0)` (hexfield never uses a non-origin center —
there is no crop center).

Mechanics at expansion (per row, symmetry chosen by the shared deterministic selector,
`hexo_train/epoch/symmetry.py`, per epoch per row — restnet's flow):

1. Transform **coordinates only**: stones (with owner + placement_index untouched), hot-cell
   lists, policy / opp-policy action ids (`transform_action_id`), first-stone cell.
2. Re-derive support, canonical `(q,r)` sort, neighbor table, and `dist_nearest_stone` **from the
   transformed coords**. Direction typing then permutes naturally — the cell that was neighbor D0
   becomes neighbor Dπ(0) of the transformed node. **No conv-weight permutation, no bias-table
   permutation, no head transformation is ever applied** (same as dense_cnn: the network is not
   constrained to be equivariant; it is trained toward it).
3. Bias indices are computed from transformed offsets — they hit different exact-table entries;
   ring/far buckets are D6-invariant by construction (hex distance is preserved); token classes
   unaffected.

Search-side: no symmetry handling (raw orientation, same as every lineage).

Tests: T5 (§9.4) asserts `expand(facts, g)` equals `expand(facts, id)` under the node bijection
induced by `g` (features permute, neighbor columns permute by the direction permutation of `g`,
policy mass follows cells); plus `inverse_index` round-trips on the packed-id basis.

---

## 6. Search integration

### 6.1 Provenance decision: extraction, not reimplementation (pick + justification)

The brief offers "fresh implementation vs extraction of shared code — pick and justify" (§5.6).
**Pick: extraction.** Grounds:

- The PUCT tree is *already* model-agnostic: `mcts_tree.rs` imports only engine/utils types,
  `threats_shared`, and `RustEvaluation` (mcts_tree.rs:18-34); it consumes `(action_id, prior)`
  pairs opaquely (brief §4). The continuous scheduler touches the model at exactly **three
  evaluator call sites** (mcts.rs:579, 1013, 1242) plus two scalar constants (mcts.rs:29). That is
  a one-trait seam.
- A from-scratch rewrite of ~4,360 lines (mcts.rs 2,715 + mcts_tree.rs 1,641) of subtle,
  forensically-tuned search (virtual-loss ordering, widening boundary cases, forced playouts,
  root-FPU-under-noise, reuse-root temperature fix, six mix_seed streams, TSS injection/override/
  guard) is the highest-risk lowest-value work in the whole project: a silent semantic divergence
  is indistinguishable from "the new architecture is worse", poisoning the competition's signal.
  Exact behavioral parity between two independent implementations is unattainable (fp tie-breaks),
  so it could not even be gated.
- The repo has done this exact move before and institutionalized it: `threats_shared` was
  extracted to be shared between dense_cnn and hexgt, and the one fork (hexgnn's threats.rs) is
  flagged as a CAUTION in the crate root (lib.rs:20-26, 36-46). Single-sourcing **is** the
  anti-fork discipline; copying the tree into hexfield's crate would be the fork the owner banned.

Scope of the extraction (new workspace crate `packages/hexo_search`, §9.1): `mcts_tree.rs` (tree),
the lockstep + continuous scheduler from `mcts.rs`, the eval cache/stats/`RustEvaluation` and
positional-prior parsing from `mcts_eval.rs` (these are byte-duplicated between dense_cnn and
hexgt already — mcts_eval.rs:102-112 vs hexgt mcts_eval.rs:35-44), the state-capsule intake
(`state.rs`, fully generic — it is the cross-cdylib engine cloning mechanism), the zero-copy
buffer pyclasses, and `threats_shared.rs` (moved; `hexo_models/rust/src/lib.rs` line 26 becomes
`pub(crate) use hexo_search::threats as threats_shared;` so every `crate::threats_shared::` path
in dense_cnn/hexgt compiles unchanged). dense_cnn keeps its pyclass entry layer, constants, and
encoder, and implements the evaluator trait; **hexgt and hexgnn are left untouched** (halted/
parked lineages keep their in-tree copies; refactoring them is churn with no consumer).

The single genericization (everything else moves verbatim):

```rust
// hexo_search::evaluation
pub struct Evaluation { pub value: f32, pub legal_action_count: usize,
                        pub priors: Vec<(PackedCoord, f32)> }
pub struct EvalRequest<'a> { pub state: &'a HexoState, pub state_hash: StateHash }

// The lineage provides ONLY "unique states -> evaluations"; dedup/coalesce/cache/
// order-preservation stay generic (moved from mcts_eval.rs:415-513 verbatim).
pub trait LeafEvaluator {
    fn evaluate_unique<'py>(&self, py: Python<'py>, states: &[&HexoState])
        -> PyResult<Vec<Evaluation>>;
}
pub fn evaluate_cached<'py>(py, eval: &impl LeafEvaluator, requests: &[EvalRequest],
    cache: &SharedEvaluationCache, stats: Option<&SharedEvaluationStats>, cache_max: usize)
    -> PyResult<Vec<Arc<Evaluation>>>;
```

Notably, **no TSS call-site parameterization is needed**: `split_tactical` already forces a
tactical cell only if it appears in the node's candidates (mcts_tree.rs:908-915). For dense_cnn
candidates are in-crop legals (graceful degradation); for hexfield candidates are the full legal
set and every tactical cell is legal within `LEGAL_RADIUS` (threats_shared.rs:34-38), so injection
becomes total automatically — the frozen-win pathology is structurally impossible and main_4's C3
frozen-win override machinery is **not ported** (§12).

Preserved semantics inventory (all unchanged because unmoved-in-meaning): batched PUCT,
prior-sorted lazy edge materialization, nucleus widening (defaults policy_mass 0.95 /
max_children 96 / min_children 2), FPU + root-FPU-zero-under-noise toggle, virtual loss, Dirichlet
root noise, root-policy temperature incl. reused-root fix and early ramp, per-ply move
temperatures, tree/subtree promotion + reuse, transposition cache keyed by
`hexo_utils::hash_state` (state_hash.rs:31-44; FIFO-bounded, default 1,048,576 entries,
constants.rs:18), PCR full/fast coin, policy-init openings (truncated-exponential ply draw), the
six deterministic `mix_seed` streams (mcts.rs:61-66), `tss_enabled` master switch (landed
2026-06-12), active_root_limit (default 1024), flush-target continuous batching with the
hold/flush/stop decision (mcts.rs:257-278).

### 6.2 Session API (hexfield._rust)

`HexfieldMctsSession(max_states)` — a thin pyclass over the shared core, exposing `search(...)`
and `run_continuous(...)` with the **same positional signatures** as Model1MctsSession
(rust_bridge.py:43-108, 111-194), including the trailing
`tss_enabled` / `root_fpu_zero_under_noise` pair. The per-move result payload and epoch
diagnostics dict are produced by the shared core and keep their exact keys
(`action_id`, `action_selection`, `visit_policy_{action_ids,weights}_bytes` (u32/f32 LE),
`visit_policy_count`, `root_prior_policy_*`, `root_value`, `visits`, `diagnostics`, plus
`pcr_full`/`policy_init` flags — mcts.rs:1874-1925; epoch dict keys mcts.rs:1089-1132). The
`on_move` Python callback contract is unchanged: return `("advance", state)`,
`("replace", key, state)`, or `None` (selfplay.py:1358-1473 is the reference consumer).

Also exported: `capabilities()`, `hexfield_featurize_states(states)` (debug/test path),
`hexfield_sample_from_state(state, game_id, turn_index, metadata)` (compact facts, §7.1), and
`hexfield_threat_analysis(state)` funneling through `hexo_search::threats::analysis_pydict` for
cross-lineage TSS parity (the dense_cnn_threat_analysis pattern, encoding.rs:69-97, minus crop
fields).

### 6.3 Evaluator payload ABI (Rust → Python), exact

Built per unique-state chunk by `hexfield/rust/src/{support,features,payload}.rs`. Rows are
**sorted by node count descending** (stable by request index) before payload assembly, so the
Python side's size-grouping is pure contiguous slicing; the generic dedup layer's slot mapping
(mcts_eval.rs:425-461, moved) already restores caller order afterwards.

PyDict keys (LE byte order; buffers are read-only buffer-protocol views, zero-copy — the
PlaneBuffer pattern, mcts_eval.rs:37-100, generalized in `hexo_search::pybuffers`):

| key | dtype | shape / layout |
|-----|-------|----------------|
| `num_graphs` | int | B |
| `total_nodes` | int | M = Σ n_i |
| `total_legal` | int | T = Σ t_i |
| `node_feat` | f16 buffer | (M, 13) row-major, canonical node order per graph, graphs concatenated |
| `node_qr` | i16 buffer | (M, 2) — `(q, r)` per node (GPU bias-index construction §3.3) |
| `node_row_offsets` | i64 buffer | (B+1,) CSR over nodes |
| `nbr_index` | i32 buffer | (M, 6) global flat indices; sentinel = M |
| `legal_index` | i32 buffer | (T,) global node indices receiving policy logits, per-graph in canonical order |
| `legal_row_offsets` | i64 buffer | (B+1,) CSR over legal entries |

Rust retains, per row, the `Vec<PackedCoord>` of action ids **in the same canonical order** as
`legal_index` and never ships them (dense pattern, brief §4).

Python returns `{"values_bytes": f32 × B, "priors_bytes": f32 × T}` — values are
`decode_binned_value` scalars in the mover's perspective; priors are per-row segment softmax over
legal logits, positional. Rust zips positionally with the retained action ids, then the **shared**
finalizer validates (finite, non-negative, unique, positive mass; exact byte counts) and
descending-sorts + normalizes (mcts_eval.rs:339-357, 515-580 — moved code, same error strings) →
`Evaluation { value, legal_action_count = T_row, priors }`. As in hexgt, `legal_action_count ==
priors.len()` — the vocabulary is the legal set, no out-of-vocabulary tail (hexgt
mcts_eval.rs:39-43 precedent).

Chunking: unique states are evaluated in chunks of `HEXFIELD_EVAL_CHUNK_STATES = 1024` (mirror of
constants.rs:17). v1 featurizes inline with rayon across states (dense pattern); the hexgt-style
featurize/forward software pipeline (hexgt mcts_eval.rs:141-200) is an M8 option if profiling
shows featurization stalls the GPU — not in the v1 contract (complexity reined in).

### 6.4 Python evaluator (`hexfield/inference.py`)

`HexfieldInference.evaluate_payload(payload)`:

1. `torch.frombuffer` every buffer (no copies), upcast indices to i64 on device, features f16 →
   model compute dtype.
2. Split rows into **size groups**: cut the (descending) node-count sequence at bucket boundaries
   `L ∈ {256, 384, 512, 768, 1024, 1536, 2048, 3072}` (×~1.4 geometric ⇒ padding waste ≤ ~40%,
   8 stable shape families); merge any group with < 16 rows into its larger neighbor. n > 3072
   falls back to ceil-to-multiple-of-512 (rare-tail, assert-logged).
3. Per group: rebase `nbr_index`/`legal_index` by the group's node offset (vectored subtraction;
   sentinel M → group sentinel), run the network: flat convs on `(M_g, C)`; for each A layer
   scatter to `(G, 8+L, C)`, compute the u8 bias index from padded `node_qr` (§3.3), gather bias,
   add the −1e4 padding-key mask, sdpa, scatter back. Pair-budget chunking of G (§10.3) bounds
   bias memory.
4. Heads → per-row values (f32) and segment-softmax priors (f32), concatenated back to payload row
   order; return bytes.

fp16 inference clone (weights f16, restnet's `fp16_model` lever) is the production mode; the same
module is the training forward (one network implementation; only AMP/eval mode and grouping policy
differ). `torch.compile` story in §10.4.

### 6.5 Self-play driver

`hexfield/selfplay.py` is a fresh write that keeps restnet's *shape*: lockstep and continuous
paths; the continuous path drives `run_continuous` with the `on_move` callback applying engine
actions, building compact samples (facts from `hexfield_sample_from_state`, policy from the
payload bytes), classifying rows by `pcr_full`/`policy_init`, writer thread per game shard, live
progress JSON every interval, finalization (z / STV-EMA / moves-left / surprise weights) on the
writer side, PCR/policy-init/temperature parameters resolved natively per slot (all preserved by
§6.1). No frozen-win override branch, no length-decay branch (§12).

---

## 7. Data pipeline

### 7.1 Shard schema — `hexfield_compact_v1`

Reuses the compact **concept** (raw facts, no tensors; one `.npz` per game + JSON sidecar;
columnar with `int64` offsets length N+1 for var-length fields — compact_io.py:1-54 pattern) with
a leaner column set. Two deliberate schema simplifications, both derived from engine truth:

- **No legal-id column.** Legality is closed-form from stones: `legal = empty ∧ hexdist≤8-of-any-
  stone` (brief §1), `Opening ⇒ {(0,0)}`. Expansion derives support from stones; a CI invariant
  test pins the derivation to `state.write_legal_moves` on random engine states. This both shrinks
  shards (~1k u32/row saved, the largest column) and removes a redundancy that could silently
  disagree with search.
- **Stones and history unified.** One column `stones = (q i16, r i16, owner u8, placement_index
  u16)` *is* the placement history (every placement is a permanent stone). Recency (ch7/8),
  opp_last_turn (ch12), and first_stone (ch5: when phase == SecondStone, the max-placement_index
  stone — validated at write) are all derived from it.

Columns:

| field | dtype | notes |
|-------|-------|-------|
| `turn_index` | i32 | per-game decision index |
| `current_player` | u8 | 0/1 |
| `phase` | u8 | 0 Opening / 1 FirstStone / 2 SecondStone |
| `value` | f32 | hard z ∈ {+1,−1} (truncated games: rows dropped, never written) |
| `moves_left` | f32 | raw decisions remaining; −1 = absent |
| `stvalue` | f32 (N,3) + `stvalue_mask` | horizons [2,6,16] |
| `stones_qroi` + offsets | i16×2 + u8 + u16, packed parallel arrays | see above |
| `own_hot`, `opp_hot` + offsets | i16 qr pairs | **stored**, not derived (window logic stays single-sourced in the engine-backed featurizer; trusted-precedent: dense stores them) |
| `policy_ids`/`policy_w` + offsets | u32 / f32 | visit policy (FULL rows only) |
| `opp_policy_ids`/`opp_policy_w` + offsets | u32 / f32 | next decision's policy; empty when next was fast/init (mask_opp_from_fast semantics) |

Sidecar JSON: `game_id`, `winner`, `truncated`, `raw_rows`, `effective_rows`, surprise stats,
`model: "hexfield"`, `schema: "hexfield_compact_v1"` — same keys the dashboard already parses
where they overlap.

### 7.2 Expansion (train-read time)

`expand_sample(facts, symmetry)` (pure Python, engine-free, process-pool workers):
D6-transform coords (§5) → derive support + canonical order + neighbor table → 13 features →
targets: policy weights mapped to legal-node positions (every target id must be a derived-legal
cell — fail-loud); opp-policy mass at off-support ids dropped with a spill counter (the
`count_spill` telemetry concept, selfplay.py:1435-1439); value/STV/ML binned per §4. Collation
(`collate.py`): flat-concat features/nbr (rebased), CSR offsets, per-bucket padded layout shared
with inference.

### 7.3 Replay window & trainer

Fresh implementation of the KataGo window semantics with `replay.py` as the semantic source:
per-game shard writes under `<run>/selfplay/`, mtime-ordered scan, tapered window (keep 300k rows,
taper 0.65), keep-prob subsampling, permuted batch-aligned output shards under
`<run>/shuffleddata/<generation>/` (window-seeding via `cp -p` keeps working). Policy-surprise
frequency weighting is applied by row duplication at finalize-time (semantic source
`materialize_policy_surprise_rows`; test oracle = import the restnet function on synthetic
inputs). Trainer implements the duck-typed contract: `select_training_samples(ctx, components,
epoch)`, `train_passes(passes, sample_window, sample_symmetries, ctx, components, epoch)`,
`close()` (pool teardown — pipeline.py:94-109 calls it). Batching is length-bucketed with the
pair-budget rule (§10.3); per-step bucket choice is sampled proportional to bucket row counts.

### 7.4 Checkpoints

`{model_state, optimizer_state, epoch, train_state, config_echo, schema_version}`. Loader returns
`{"status": "loaded", "epoch": N}` for full resume (the pipeline fast-forwards to N+1 —
load-bearing contract, hexo_train/epoch/loop.py:147-166) or `{"status": "initialized", ...}` for
weights-only `initialize_from`; saver writes `epoch_NNNNNN.pt` + final `latest` pointer
(checkpoints.py:91-118 shapes).

---

## 8. Bootstrap

**BC prefit** — `scripts/bootstrap_hexfield_hf.py`, the proven 3-stage shape
(scripts/bootstrap_dense_cnn_restnet_hf.py:1-33): (1) CONVERT: replay each of the 6,902 decisive
HF corpus games (`timmyburn/hexo-bootstrap-corpus`, raw move lists) through `hexo_engine`,
one-hot policy at the human move, z from winner (= last mover; cross-checked vs engine terminal),
STV from outcome-only EMA, write **production** hexfield shards via the production writer
(no-poison property: same writer as self-play); `--validate N` mode replays without writing.
(2) PREFIT: production shuffle + `trainer.train_passes` for a few passes (batch by pair-budget).
(3) Save `{model_state, optimizer_state, epoch: 0}` + strict re-load verification. ≈ 431k
positions; support sets in human games are small (compact play) ⇒ cheap pass.

**Optional distillation** from existing restnet self-play shards: a read-only column reader for
their documented compact layout (compact_io.py) — *not* code reuse, a data-format consumer. Key
fix enabled by §7.1: their stored `legal_action_ids` are crop-restricted (sample_gen.rs:79-89),
but hexfield expansion derives **full** legality from stones, so even crop-clipped marathon rows
re-expand correctly; policy/value targets transfer as-is (action ids are universal). Gated as
M7-optional — BC prefit alone is the committed warm start.

---

## 9. Code architecture (the lens)

### 9.1 Crate & package layout

```
Cargo.toml (workspace)            members += packages/hexo_search, packages/hexfield

packages/hexo_search/             NEW shared crate (rlib only; no cdylib)
  Cargo.toml                      deps: hexo_engine, hexo_utils; pyo3 optional behind "python"
  src/lib.rs
  src/threats.rs                  MOVED from hexo_models/rust/src/threats_shared.rs (items pub)
  src/seeds.rs                    mix_seed + 6 stream tags + random_unit        [no pyo3]
  src/evaluation.rs               Evaluation, EvalCache (FIFO-bounded), Stats,
                                  evaluate_cached + LeafEvaluator trait,
                                  positional parse + finalize_priors            [core no-pyo3;
                                                                                 PyErr conv under "python"]
  src/tree.rs                     MOVED mcts_tree.rs                            ["python"]
  src/scheduler.rs                MOVED lockstep+continuous core of mcts.rs     ["python"]
  src/state_intake.rs             MOVED state.rs (engine state capsule)         ["python"]
  src/pybuffers.rs                zero-copy read-only buffer pyclasses
                                  (F16/I16/I32/I64/F32 views)                   ["python"]

packages/hexo_models/             live host crate — minimal touch
  rust/src/lib.rs                 line 26: `pub(crate) use hexo_search::threats as threats_shared;`
  dense_cnn/rust/src/mcts_tree.rs DELETED (moved)
  dense_cnn/rust/src/mcts.rs      shrinks to: pyclass Model1MctsSession + arg parsing +
                                  Model1Evaluator (impl LeafEvaluator over encode_model1_*)
  dense_cnn/rust/src/mcts_eval.rs keeps encoder payload + PlaneBuffer; cache/parse delegate
  hexgt/, hexgnn (sibling)        UNTOUCHED (halted/parked; their copies stay; documented)

packages/hexfield/                NEW lineage — all fresh code
  Cargo.toml                      deps: hexo_engine, hexo_utils, hexo_search (python), half,
                                  ahash, rayon, pyo3(workspace)
  pyproject.toml                  maturin; module-name = "hexfield._rust";
                                  python-source = "python"; features = ["python"];
                                  entry point: [project.entry-points."hexo_train.models"]
                                  hexfield = "hexfield.plugin:get_plugin"
  rust/src/lib.rs                 #[pymodule] _rust: session, featurize, sample, capabilities
  rust/src/support.rs             support set + canonical order + nbr table
  rust/src/features.rs            13 features (f32) + f16 conversion
  rust/src/payload.rs             §6.3 payload assembly (size-desc sort, buffers)
  rust/src/evaluator.rs           HexfieldEvaluator: impl LeafEvaluator (chunk, encode, call,
                                  shared parse)
  rust/src/session.rs             #[pyclass] HexfieldMctsSession → hexo_search scheduler
  rust/src/sample.rs              hexfield_sample_from_state
  python/hexfield/
    constants.py config.py d6.py geometry.py     (LUT enumeration, hexdist, packing)
    support.py features.py                       (Python twins for expansion)
    collate.py architecture.py losses.py
    inference.py rust_bridge.py
    selfplay.py samples.py compact_io.py replay.py
    trainer.py checkpoints.py plugin.py evaluation.py performance.py player.py

configs/hexfield_main_1.toml      [model] name="hexfield", module="hexfield.plugin"
scripts/_rebuild_hexfield.sh      maturin develop --release -m packages/hexfield/Cargo.toml
                                  --features python   (hexgt-build venv pattern,
                                  _rebuild_hexo_models_hexgt.sh:1-23)
scripts/bootstrap_hexfield_hf.py
tests/test_hexfield_*.py, tests/test_hexo_search_*.py
```

Module names deliberately mirror the restnet decomposition (10,823 lines across 25 modules,
proven) so repo navigation transfers; every file is a fresh write.

Dependency hygiene: hexfield's Python package depends on `hexo-engine`, `hexo-runner`,
`hexo-train`, `torch`, `numpy` — **not** on `hexo-models` (the old model zoo is not imported at
runtime; only tests import it as an oracle). hexo_search has no torch and no engine-mutation
surface. No `#[path]` escapes the package directory anywhere (the hexgnn CAUTION class,
lib.rs:41-46, is structurally avoided).

### 9.2 Rust/Python boundary (one sentence per crossing)

1. **States in:** Python hands live `hexo_engine.HexoState`s; `hexo_search::state_intake` clones
   them via the versioned C-ABI capsule (state.rs:20-101, moved verbatim).
2. **Eval out/in:** Rust ships the §6.3 dict (zero-copy buffers); Python returns
   `values_bytes`/`priors_bytes`; shared finalizer validates/sorts/normalizes.
3. **Moves out:** the shared scheduler invokes Python `on_move(game_key, payload)` per decided
   move; Python returns advance/replace/None (selfplay.py:1358-1473 contract, unchanged).
4. **Facts out:** `hexfield_sample_from_state` returns the compact facts dict (stones with
   placement indices, hot lists, phase, metadata) for the recorder.

### 9.3 Build/rebuild story alongside the live main_4 run

Hard property: **building hexfield can never change main_4's behavior.**

- `hexfield._rust` is its own cdylib; `maturin develop` for hexfield writes only hexfield's
  package — `hexo_models._rust`'s installed `.so` is not rebuilt, reinstalled, or reloaded
  (additive-install precedent: dense_cnn_restnet's pyproject makes the same guarantee in prose,
  pyproject.toml:11-16).
- The hexo_search extraction edits dense_cnn **source**, which is inert until someone runs
  `scripts/_rebuild_hexo_models_hexgt.sh` (lib.rs:16-18; HANDOFF.md: "Rust edits are inert until
  that script runs"). The supervisor never rebuilds.
- Discipline encoded in the milestone gates: all M0 verification builds install into a separate
  dev venv (`/root/.venvs/hexfield-dev`), never the live `hexgt-build` venv — because the live
  supervisor *relaunches the training process between epochs* and would pick up a replaced `.so`
  from disk. The live venv adopts the (gate-proven, byte-identical) rebuilt hexo_models only at an
  owner-chosen supervisor halt point — their existing mechanic.
- Cargo.lock: new crates only add entries (workspace-pinned versions), so a later dense_cnn
  rebuild resolves the exact same dependency versions as today.
- The extraction lands as its own revertable commit train, fully before any hexfield code.

### 9.4 Parity oracles & test strategy

Tests live flat in `tests/` (repo convention), authoritative in the WSL venv. The trusted lineage
is imported **as oracle, never as dependency**:

| id | test | oracle / gate |
|----|------|---------------|
| T0 | `test_hexo_search_extraction_golden.py` — lockstep `search` + `run_continuous` over scripted multi-game batches with a deterministic stub evaluator (priors/values = splitmix of state_hash), fixed seeds, all levers exercised (PCR, policy-init, ramp, TSS on/off, noise, reuse) | golden JSON recorded against the PRE-extraction `.so`; post-extraction run must match **exactly** (chosen actions, visit counts, move-class tallies, diagnostics). Plus the existing suites must stay green: test_dense_cnn_continuous_scheduler, _temperature_schedule, _restnet_pcr_policy_init, _dense_cnn_tss, _restnet_tss_fpu_toggle |
| T1 | `cargo test -p hexo_search` (default features) | moved threats tests + seeds/eval-cache/widening/flush-decision unit tests, no Python needed |
| T2 | featurizer parity: random engine games, every ply | Rust `hexfield_featurize_states` == Python `expand(sample_facts, identity)` — bit-equal after both cast f32→f16 |
| T3 | support invariants I1/I2/I3 + legality-derivation == `write_legal_moves` on random states | engine is the oracle |
| T4 | payload round-trip: crafted Python stub returns known priors; assert Evaluation ordering/normalization; corrupt-byte cases raise the exact shared error strings | shared finalizer semantics (mcts_eval.rs:515-580) |
| T5 | D6 expansion consistency (§5) + `inverse_index` round-trip | self-consistency under node bijection |
| T6 | HexNodeConv ≡ dense_cnn `HexConv2d` on embedded patches (§3.1); missing-neighbor zero semantics | import `dense_cnn_restnet.architecture.HexConv2d` as executable spec |
| T7 | attention: sdpa vs materialized oracle ≤ 1e-5 (fp32); u8 bias-index vs brute-force Python classifier over random coord sets incl. token rows; padding-mask no-leak (padded rows perturbed ⇒ real outputs unchanged) | restnet's proven oracle pattern (architecture.py:271-280) |
| T8 | 65-bin helpers + STV-EMA + surprise weighting on synthetic sequences | import restnet `losses`/`samples`/`replay` functions as oracles |
| T9 | segment softmax / segment CE vs per-row loop | brute force |
| T10 | E2E self-play smoke: tiny net, 8 games, both schedulers; rows written, schema-valid, deterministic across two same-seed runs; PCR/init flag accounting | self-consistency + flag math |
| T11 | plugin pipeline: `TrainingPipeline().run()` micro-config (CPU, 1 epoch, 2 games); resume from epoch-N checkpoint fast-forwards to N+1 | hexo_train contract (loop.py:147-166) |
| T12 | bootstrap `--validate`: replay 20 corpus games, legality/terminal/winner | engine |

### 9.5 Plugin contract integration

`hexfield/plugin.py` implements the full duck-typed surface hexo_train dispatches on
(registry.py:20-25): `name`, `build_model(game_spec, config)`,
`training_component_overrides(...)` returning `ComponentOverrides(trainer=…, optimizer=…,
checkpoint_loader/saver=…, uses_shared_sample_store=False, extra={…})` (components.py:60-84),
plus `generate_selfplay`, `evaluate_epoch`, `calibrate_performance`. Config mode 1
(`[model] module = "hexfield.plugin"`) works PYTHONPATH-only before install (registry.py:64-65);
the entry point covers installed mode. `evaluate_epoch` plays G lockstep games vs an anchored
reference-checkpoint ladder (restnet evaluation-stage shape) and writes the same JSON families the
dashboard reads; `player.py` provides a `hexo_runner.player.RunnerPlayer` adapter so sealbot/arena/
match-screen workers can host hexfield checkpoints (lineage tag `"hexfield"` flows from the run
manifest automatically).

---

## 10. Perf budget

### 10.1 Parameters (exact)

| component | count |
|-----------|-------|
| stem 7·13·96 + BN | 8,928 |
| 6 conv blocks × (2×(7·96² + BN)) | 776,448 |
| 3 A blocks × (QKVO 37,248 + LN 384 + MLP 37,152 + bias 229·4) | 227,100 |
| token init 8×96 | 768 |
| policy + opp heads (7·96²+96 conv, +97 linear) ×2 | 129,410 |
| value head 192→64→65 | 16,577 |
| aux body 192→64 + 4 tops 64→65 | 29,252 |
| **total** | **≈ 1.188 M** |

### 10.2 FLOPs/eval vs dense_cnn (96ch restnet `R_R_R_T_R_R_T_R` @ 41², ≈ 6.6 GFLOPs/eval)

Per node: conv-side ≈ 1.82 MF (stem 17.5k + 12 trunk convs ×129k + 2 head convs); per A layer
147k + 384·L (scores 4·L·C). Totals (+ ~15% padding to bucket):

| support N (L bucket) | GFLOPs/eval | × dense |
|---------------------|------------:|--------:|
| 271 (320)  | 0.79 | 0.12 |
| 600 (640)  | 1.86 | 0.28 |
| 1000 (1024) | 3.56 | 0.54 |
| 1500 (1536) | 6.2 | 0.94 |
| 3000 (3072) | 17.8 | 2.7 |

Typical self-play positions (600–1500) are cheaper than the dense crop; the cost crossover sits at
N ≈ 1550; the ≥2k tail is more expensive — bounded by group-batching and rarer under healthy
(non-marathon) play. Per-eval host work (support build + nbr table) is O(N) hash ops, well under
dense's 21,853-float plane stamping.

### 10.3 Memory & batching (the one new mechanism)

Attention bias dominates: fp16 `(G, 4, (8+L)², )` ⇒ per graph 2.1 MB @L=512, 8.5 MB @1024,
34 MB @2048, 76 MB @3072. **Pair-budget rule** (single policy, used by both inference grouping and
trainer batching): `G_max(L) = clamp(floor(PB / L²), 4, 64)` with PB = 33.5M pairs (≈ batch-32 @
L=1024 parity). Inference additionally caps transient bias at 256 MB per chunk. Training VRAM @
pair-budget: bias ≈ 3 layers × 270 MB (fp16, saved for the table gradient) + activations
(≈ 33k nodes × 96 × ~40 tensors) ≈ 250 MB + optimizer states (tiny model) ⇒ ≈ 1.5–2 GB — same
class as restnet's batch-32 footprint on the shared 12 GB card. Inference @ ~256 leaves: grouped
forwards, ≤ 256 MB transient bias + ≤ 150 MB activations + f16 weights (2.4 MB).

GEMM shapes vary with M but matmuls don't pay the cuDNN-benchmark re-autotune tax (that trap is
conv2d-specific; we have no conv2d). The ~8 attention bucket lengths bound the distinct attention
shapes.

### 10.4 fp16 / compile / TRT story

fp16: f16 feature transport (load-bearing per brief) + fp16 inference weights, tolerance-gated.
`torch.compile`: per-bucket compilation with `dynamic=True` on the flat M dim — ≤ 8 specializations,
applied only to the frozen inference clone (restnet's compile lessons: frozen bias/no
data-dependent guards); ships OFF by default, M8 gates it. TRT: **not in v1** — the gather/scatter
+ dynamic-shape graph is exactly the export class the brief flags as may-not-export; stated
fallback is sdpa + compile + bucketing. FlexAttention (score_mod computing bias from coords,
removing the O(L²) bias materialization entirely) is recorded as the designated future perf lever,
behind the same tolerance gate — not a v1 dependency.

---

## 11. Milestones (each independently verifiable; GPU scheduling out of scope)

| # | deliverable | acceptance gate |
|---|-------------|-----------------|
| M0 | `hexo_search` extraction + dense_cnn shims (separate commit train; no hexfield code) | T0 golden byte-determinism in the **dev venv**; full existing dense_cnn/restnet pytest set green; `cargo test -p hexo_search`; `cargo build -p hexo_models --features python`; live venv untouched |
| M1 | hexfield skeleton: crate + pyproject + plugin + config + rebuild script | plugin resolves via module AND entry point; `build_model` constructs the network; synthetic NodeBatch forward returns all head shapes; param count == §10.1 ± renames |
| M2 | support/featurizer (Rust) + expansion (Python) + compact_io + sample facts | T2, T3, T5; shard write/read round-trip |
| M3 | network + losses | T6, T7, T8, T9; loss decreases on an overfit-one-batch probe |
| M4 | evaluator boundary + session (lockstep first) | T4; lockstep self-play smoke (CPU tiny-net) produces legal games; cache hit-rate counters sane |
| M5 | continuous scheduler wiring + selfplay driver + writer + finalization | T10; flush/move-class diagnostics keys present; spill + drop-truncated accounting |
| M6 | trainer + replay window + checkpoints + evaluate_epoch | T11; 2-epoch CPU micro-run incl. resume; shuffle dir layout matches §7.3 |
| M7 | BC bootstrap (+ optional restnet-shard distillation reader) | T12; prefit CE falls ≥ 30% over passes; strict re-load; (optional) reader re-expands 1k restnet rows with zero legality violations |
| M8 | perf: fp16 clone, calibration hook, bucket tuning, (opt) compile, (opt) eval pipeline | `calibrate_performance` selects batch sizes; fp16-vs-fp32 value/prior tolerance gate; GPU throughput probe vs dense baseline recorded (informational, no target) |
| M9 | first real run readiness: `hexfield_main_1.toml`, supervisor script, dashboard sidecars, runner adapter | end-to-end 1-epoch GPU run in the dev venv writes complete artifacts; arena adapter plays a full game vs a restnet checkpoint |

Ordering rationale: M0 is isolated and maximally reviewable; M1–M3 are GPU-free and parallelizable
with M0 review; search lands lockstep-before-continuous so the eval boundary is debugged on the
simpler scheduler; every gate is runnable by one command in the dev venv.

---

## 12. Envelope deviations (all from §3 defaults; no §2 violations found)

1. **hot-window threshold:** brief §3 text says "≥ count-3"; I port the trusted dense_cnn
   semantics — **single-colour count ≥ 4, gated `placements_made ≥ 7`** (encoding.rs:232-249).
   "Port of the 13 trusted planes" is the operative clause; ≥4 is also the Connect6-meaningful
   threshold (count-4 + two placements = win-in-one-turn). Flagged loudly: if "≥3" was intended,
   it is a one-constant change in two featurizers + tests.
2. **Shard schema deltas** (sanctioned by §5.7 "or define new"): legal ids, first_stone, and a
   separate history column are **derived, not stored** (§7.1); phase stored as u8 not string.
3. **dist_nearest_stone normalization:** /8 with halo at 1.125 (>1 by design); 0-fill on the
   empty board.
4. **opp_policy support:** logits restricted to this position's legal node set with spill-drop +
   telemetry (dense trains it unmasked on the full crop; a crop does not exist here).
5. **Per-A-block bias tables** (3 × 229×4) rather than one shared table — matches restnet's
   per-block precedent; +1.8k params.
6. **Bias-index mechanism** (u8 LUT, on-GPU construction, −1e4 fp16-safe mask) and the
   **pair-budget bucketed batching** are Claude-added mechanisms — each isolated to one module and
   oracle-tested.
7. **Not ported:** frozen-win override (C3) and length-decay (C1) — both are crop-pathology
   mitigations; the pathology is impossible by construction here (§6.1). `drop_truncated_rows =
   true` IS adopted as the default.
8. **STV loss weight 0.1** (brief §4 production) though restnet's code default is 0.25 — set in
   config like the live runs do.
9. **Norm:** BN over flat support nodes is a semantic shift vs BN over the fixed crop
   (population = real cells only); `norm = "layer"` fallback knob retained.
10. **TRT excluded from v1; compile optional** (§10.4) — the brief asks for a stated story, this
    is it.
11. **Search provenance = extraction** of dense_cnn's search core into `hexo_search` with
    dense_cnn single-sourced onto it (the §5.6 sanctioned alternative to a fresh implementation),
    chosen over rewrite for the reasons in §6.1; hexgt/hexgnn intentionally not migrated.
