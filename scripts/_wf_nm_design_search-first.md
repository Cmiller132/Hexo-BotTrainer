# hexspan — a search-economics-first native-hex lineage

Design competition entry. Lens: **SEARCH ECONOMICS** — the model is designed backward from the MCTS
evaluator boundary on a single 12GB RTX 4070 Ti shared with training. Every choice below is priced in
bytes, FLOPs, kernel shapes, and evals/sec before it is priced in anything else.

---

## 1. Identity

**hexspan** is a stone-anchored, variable-N native-hex network whose unit of compute is the *support
node* (legal ∪ stones ∪ 1-ring halo), not a crop pixel. The thesis: dense_cnn/restnet pay a fixed
1,681-cell toll per eval (including 420 permanently-dead corner cells and, mid-game, hundreds of cells
nobody can play), and bought that toll's GPU-friendliness at the price of the radius-20 crop that
killed main_3. hexspan makes the *engine's own legal set* the tensor: payload bytes, Rust featurize
time, conv FLOPs, and attention pairs all scale with the position's true size (217→~3,000 nodes),
with zero coverage loss by construction. The GPU-friendliness is recovered not in the model but in the
*evaluator*: a sort-by-size, fixed-shape packing layer turns arbitrary staggered-root flushes into at
most 7 static (batch, length) kernel shapes, so cuDNN/compile/CUDA-graph economics survive variable N.
Trunk = 6 direction-typed 7-tap conv blocks + 3 attention blocks with 8 bidirectional summary tokens
(`C C C A C C A C A`), 96 channels, **1.232M params**, ~2.9 GFLOPs at the mid-game median vs restnet's
~5.8 GFLOPs always. Search semantics (PUCT, nucleus widening, PCR, policy-init, continuous + lockstep
schedulers, TSS, state-hash eval cache) are re-implemented from scratch in a greenfield crate and
pinned to the proven dense_cnn behavior by golden tests.

---

## 2. Input representation

### 2.1 Support set

For a non-terminal engine state `s` (all sets of axial `(q, r)` i16 coords):

```
L  = engine legal set            (write_legal_moves; empty cells within hex-distance 8 of any stone,
                                  hexo_engine/rust/src/legal.rs:18 LEGAL_RADIUS = 8)
St = stones                      (board().occupied_cells())
H  = halo = empty cells not in L that are hex-adjacent (distance 1) to any cell of St ∪ L
SUPPORT = L ∪ St ∪ H,  N = |SUPPORT|
```

**Halo closed form (used by the encoder and validated by a property test):** for a non-empty board,
`H = { c : min over stones t of hexdist(c, t) = 9 }` — the distance-9 ring. Proof: a cell adjacent to
a stone has d ≤ 1 → in L if empty; a cell adjacent to a legal cell has d ≤ 9, and if d ≤ 8 it is in
L (or a stone) itself, so halo cells have d = 9 exactly. Conversely hex distance is a path metric, so
every d = 9 cell has a d = 8 neighbor, which is empty (d > 0) hence legal. ∎
Halo nodes carry features but **never logits** (owner lock #1).

**Ply-0 / empty-board edge case** (Opening phase): engine legal = {(0,0)} only
(hexo_engine/rust/src/state.rs:223-228). `SUPPORT = {(0,0)} ∪ ring1(0,0)`, N = 7, L = 1, St = 0,
H = 6. `dist_norm` (feature 11) is defined as 0 everywhere on this one state (no stones exist).
The encoder is total; whether the driver searches the forced move or shortcuts it is scheduler policy
(unchanged from dense_cnn: it searches it; the tree has one edge).

**Size envelope** (drives every budget below): |disk(8)| = 1 + 3·8·9 = 217.
k=1 stone → N = 1 + 216 + 54 = 271. Mid-game ≈ 600–1,500. Long spread games ≈ 3,000 (brief §4).
Hard cap: encoder errors loudly at N > 65,534 (u16 neighbor indices, §6.2); theoretical max with
max_actions=1024 placements is ~10⁵ but self-play has never exceeded ~3k.

### 2.2 Node ordering (layout contract)

`SUPPORT` is serialized as **[ legal | stones | halo ]**, each segment sorted ascending by
`PackedCoord` u32 — which *is* ascending `(q, r)` order by construction
(hexo_engine/rust/src/legal.rs:20-35: the packing preserves signed (q,r) sort order). Consequences:

- **Legal-prefix property:** the legal nodes of graph g are exactly slots `[0, L_g)`. The evaluator
  payload needs no legal-index arrays at all (dense_cnn ships i64 flats — 8 bytes/legal move,
  mcts_eval.rs:286-298; we ship one i32 count per graph), and priors return positionally over the
  prefix. Rust keeps the sorted `Vec<PackedCoord>` it wrote and zips positionally (same proven
  pattern as mcts_eval.rs:356-390).
- Ordering is pure layout: convs consume explicit neighbor indices and attention consumes coords, so
  the model is permutation-equivariant given those arrays; D6-augmented rows simply re-sort.

### 2.3 Node features — 13 channels, f16 on the wire, exact formulas

`latest = placements_made()`. All features ∈ [0, 1.125].

| # | name | formula | extent |
|---|------|---------|--------|
| 0 | own_stone | 1 if stone of current player | stones |
| 1 | opp_stone | 1 if stone of opponent | stones |
| 2 | empty | 1 − f0 − f1 | legal + halo |
| 3 | legal | 1 if node index < L_g | legal |
| 4 | phase_second | 1 on **all** nodes iff phase == SecondStone | const plane |
| 5 | first_stone | 1 at the turn's first placement cell (SecondStone only; an own-stone node) | 1 cell |
| 6 | player_colour | 1 on all nodes iff current player == Player0 | const plane |
| 7 | own_recency | own stones: `1 / (1 + (latest − placement_index))`; else 0 | stones |
| 8 | opp_recency | same, opponent stones | stones |
| 9 | own_hot | 1 if node is an **empty** cell of an active single-colour window with `count(owner) ≥ HOT_COUNT_MIN` owned by current player | empties |
| 10 | opp_hot | same, opponent-owned windows | empties |
| 11 | dist_norm | `hexdist_to_nearest_stone / 8.0` (stones → 0, legal ≤ 1.0, halo = 1.125; ply-0 = 0) | all |
| 12 | opp_last_turn | 1 at the 1–2 cells of the opponent's most recent completed turn (dense_cnn semantics, dense_cnn/rust/src/encoding.rs:268-298) | ≤2 cells |

`HOT_COUNT_MIN = 3` (brief §3 default; note dense_cnn production used ≥4 with a placements≥7 perf
shortcut, encoding.rs:226-238 — the threshold is a config knob, see §12.4). Hot is computed by one
pass over `board().windows().entries()` filtering `active_player() == Some(p) && count(p) ≥ 3`,
marking the entry's empty cells; no placement-count gate (the gate was only a scan shortcut).

f16 is loss-free here: 9 features are binary, dist_norm is k/8, recency is 1/(1+age) — all within
f16's exact or near-exact range; dense_cnn already ships f16 planes under a byte-identical gate
(mcts_eval.rs:44-47).

**Single-featurizer rule:** one Rust featurizer with two entry points — `encode_state(&HexoState)`
(search) and `encode_facts(stones, history, legal_ids, phase, first, player)` (training expansion,
windows/hot/dist recomputed from stones) — pinned bit-equal by a parity test (§9). No Python
featurizer exists; the dense_cnn dual-implementation drift surface (encoding.rs vs input.py) is
deleted.

---

## 3. Trunk

Interleave **`C C C A C C A C A`** (9 layers — accepted default), channels **C = 96**, heads **4**
(head_dim 24), mlp_ratio **2**.

### 3.1 Stem

`x = ReLU(MaskedBN(DirConv7(feat)))`, DirConv7: 13→96, bias-free (BN supplies bias).
Embeds + 1 hop of context, so block C1 starts with radius-1 features.

### 3.2 Direction-typed conv (the primitive, owner lock #3)

Canonical direction order (documented contract, CCW from +q):
`DIR = [(+1,0), (+1,−1), (0,−1), (−1,0), (−1,+1), (0,+1)]`, slot 0 = self.

```
DirConv(x)[i] = b + Σ_{d ∈ {self} ∪ DIR}  W_d · x̂[nbr_d(i)]      W_d ∈ R^{C_out×C_in}
x̂[j] = x[j] if j is a valid support node else 0                    (= conv zero-padding semantics)
```

**GPU realization — one GEMM:** gather `xg = x[nbr]` → `[B, N, 7, C]`, multiply by the {0,1} validity
mask `[B, N, 7, 1]` (this *is* the zero-padding), reshape `[B·N, 7C]`, one matmul `@ W[7C, C]`.
K = 672 is a healthy GEMM reduction dim; no cuDNN, no 925ms autotune events (matmul uses cuBLAS
heuristics), no 9-tap waste (dense_cnn computes all 9 square taps with 2 masked to zero,
restnet architecture.py:163-179 — we compute exactly 7).

### 3.3 C block (dense_cnn family, post-activation residual)

```
y = ReLU(MaskedBN1(DirConv7_1(x)))
y = MaskedBN2(DirConv7_2(y))
x = ReLU(x + y)
```

**Norm = masked BatchNorm, and why (search-econ):** at inference BN folds into the adjacent DirConv
GEMM (`W ← γ/σ·W`, fused bias) — *zero* runtime ops, exactly the dense_cnn folding story
(restnet architecture.py:914-921). LayerNorm would cost two reductions per node per layer forever and
cannot fold. Train-time masked statistics: `μ_c = Σ(x·m)/Σm`, `σ²_c` likewise, where m excludes
padding rows and token slots; running stats updated with masked counts (≈15 lines, autograd-clean;
batch 32 × ~900 valid nodes ≈ 29k samples/step — statistics are healthy). Config
`conv_norm = "bn" | "ln"` keeps LN as the fallback knob (default "bn").

### 3.4 A block (owner lock #5: pre-norm transformer, learned rel-pos bias)

Token tensor layout from A1 onward: `x = concat(tokens[8], cells[N_pad])` → `[B, S, C]`,
`S = N_pad + 8`, **tokens at slots [0, 8)**, cells at `[8, 8 + N_pad)` (legal prefix = slots
`[8, 8 + L_g)`). C blocks after A1 process `x[:, 8:]` and re-concat (tokens are *carried through*
unchanged — lock #6).

```
u = LN1(x);  x = x + Wo·SDPA(u·Wq, u·Wk, u·Wv, attn_mask = BIAS)     # 4 heads, scale 1/√24
v = LN2(x);  x = x + MLP(v)                                          # Linear 96→192, GELU, 192→96
```

`impl = "sdpa"` default with a materialized-scores oracle sharing the same parameters for tests —
restnet's proven dual-path pattern (architecture.py:262-533).

### 3.5 Relative-position bias (re-keyed on hex offsets, unbounded-safe)

**One shared table for all 3 A layers** (deviation §12.1): `Parameter[301, 4]`, zero-init
(restnet zero-inits its bias table, architecture.py:319-321).

Index of pair (query i, key j), with `Δq = q_i − q_j`, `Δr = r_i − r_j`,
`d = max(|Δq|, |Δr|, |Δq+Δr|)`:

| condition | index | entries |
|---|---|---|
| both cells, d ≤ 8 | `(Δq+8)·17 + (Δr+8)` | 289 slots (217 reachable; 72 dead — §12.2) |
| both cells, 9 ≤ d ≤ 16 | `289 + (d − 9)` | 8 ring buckets |
| both cells, d ≥ 17 | `297` | 1 far bucket |
| query token, key cell | `298` | 1 |
| query cell, key token | `299` | 1 |
| token, token | `300` | 1 |

Token entries are per-head scalars — exactly lock #6's "per-head learned bias scalar vs cells".
Padded cell **key columns** additionally receive `PAD_BIAS = −30000.0` (finite in fp16; `exp → 0`
exactly; avoids restnet's −1e9→−inf saturation note, architecture.py:90-97). **No row can be fully
masked by construction:** token keys (slots 0–7) are never masked, so even padding-query rows softmax
over ≥ 8 finite entries — the NaN hazard class is structurally absent.

**Build & cost (the one data-dependent tensor):** per packed forward, from padded coords
`[B, S, 2]` i32: compute the index matrix (elementwise int ops), gather the 2.4KB table (L1-resident)
into `BIAS [B, 4, S, S]` f16, overwrite the 8 token rows/cols with the 3 constants via static slices,
add PAD_BIAS columns. Built **once**, reused by all 3 sdpa calls (the shared-table payoff). Unlike
restnet, this bias is per-graph (coords differ), so it can never be frozen/cached across forwards —
its traffic is priced in §10.3 as the design's #1 known inefficiency, with the FlexAttention escape
hatch (§10.5).

### 3.6 Per-layer rationale (accepted default, restated against this geometry)

- **C1–C3 first**: stem+3 blocks = receptive radius 7 ≥ the 5-step span of a 6-window — every node
  knows its full local line context before any global mixing.
- **A1**: tokens join; first global aggregation of digested line features (the anti-crop layer: rim
  and out-of-rim activity meet here).
- **C4–C5**: local re-resolution under global context.
- **A2**: completes the first full cell→token→cell round trip (hub functional — lock #6 needs ≥2 A's).
- **C6**: final local sharpening for policy geometry.
- **A3 last**: heads read maximally fresh tokens; policy gets a final global-urgency pass.

### 3.7 Init

DirConv weights: PyTorch Linear default (kaiming-uniform over fan-in 7C). All other Linears:
`trunc_normal_(std=0.02)`; LN/BN weight 1 bias 0 (restnet's ViT-style init, architecture.py:876-885);
tokens `trunc_normal_(0.02)`; bias table zeros.

---

## 4. Heads & losses

Five heads (lock #8; **no spatial ownership/win-window head**). Token split (accepted default):
tokens 0–1 → main value, tokens 2–3 → aux (STV + moves-left), tokens 4–7 uncommitted hub.

| head | ops | output | search? |
|---|---|---|---|
| policy | `DirConv7(96→96, bias) → ReLU → Linear(96→1)` on cell slots | `[B, N_pad]` logits | **yes** |
| opp_policy | same structure, separate params | `[B, N_pad]` | no |
| value | `concat(tok0, tok1) [192] → Linear(192→128) → ReLU → Linear(128→65)` | 65-bin | **yes** (decoded scalar) |
| stvalue_{2,6,16} | aux trunk: `concat(tok2, tok3) → Linear(192→128) → ReLU` (shared) → per-horizon `Linear(128→65)` | 65-bin ×3 | no |
| moves_left | aux trunk → `Linear(128→65)` | 65-bin | optional (§6.5) |

**Pathway separation (heads_v3 lesson, natively):** the main value MLP reads *only* tokens 0–1; the
aux MLP reads *only* tokens 2–3. Non-stationary aux targets cannot compete with the game-outcome head
for readout capacity — separation by token, not just by reduction module.

**Targets / masking / weights** (production values, brief §4 + configs/dense_cnn_restnet_main_4.toml:120-134):

- `policy` (w=1.0): MCTS visit distribution with forced-playout pruning (preserved semantics),
  expanded to mass on **legal-prefix slots**; soft-CE masked to the legal prefix (the native analogue
  of restnet's legal-masked CE, losses.py:56-97). Halo/stones get no logits gradient — lock #1 enforced
  structurally by the mask.
- `value` (w=1.0): hard-z outcome ∈ {−1, 0, +1} → adjacent-bin soft target on 65 bins
  (`scalar_to_binned_target` semantics, losses.py:33-53).
- `opp_policy` (w=0.25): the recorded next-decision policy (restnet semantics), **projected** onto
  this row's legal slots — pairs whose action id is not in this position's legal set are dropped and
  the row renormalized; zero-mass rows allowed (`allow_zero_rows=True`, matching losses.py:158).
  Dropped-mass counter exported per epoch (deviation §12.10; restnet instead left the head unmasked on
  its 1681 crop, losses.py:147-153 — natively there is nothing outside support to project onto).
- `stvalue_h, h ∈ {2, 6, 16}` decisions (w=0.1 each): EMA targets carried in shard rows (restnet
  finalize semantics unchanged); per-row mask.
- `moves_left` (w=0.1): remaining decisions clamped to `[0, MOVES_LEFT_CAP=512]`
  (dense_cnn_restnet/constants.py:27), mapped linearly to [−1, 1], 65-bin soft target, masked rows
  (truncated games / absent).

---

## 5. Symmetry — D6 by training-time augmentation (lock #9)

All augmentation is **fact-level**, before featurization, with the transform center always the origin
(no crop → no per-sample center, a strict simplification vs dense_cnn's center-relative transforms):

1. Draw symmetry index `σ ∈ [0, 12)` per row per epoch (pipeline-supplied, as today).
2. Transform every coordinate-bearing fact with the axial D6 maps (same math as restnet d6.py:129-136:
   rot60 `(q,r) → (−r, q+r)`, reflect `(q,r) → (q, −q−r)`): stones, placement history, legal action
   ids, policy/opp-policy action ids, first_stone, opp_last_turn cells.
3. Re-sort segments by PackedCoord, rebuild support/halo/nbr/features via the **same** Rust
   facts-featurizer.

Consequences requiring zero model-side machinery:

- **Directions:** neighbor `d`'s content rotates with the data (nbr arrays are rebuilt from
  transformed coords); per-direction conv weights `W_d` are *not* permuted and learn approximate
  equivariance statistically — exactly the trusted dense_cnn approach mapped to direction-typed ops.
- **Bias tables:** indexed by *transformed* offsets; never transformed themselves; learn approximate
  D6 symmetry from data. (D6 permutes the 217 exact-offset entries and fixes ring/far/token entries.)
- **Hot/threat facts:** recomputed in the transformed frame; windows map to windows under D6
  (threats_shared.rs:40-41).
- **Exactness:** there is no representational boundary, so augmentation can never spill or drop a row
  (dense_cnn's disk-guard/drop machinery, compact_io.py:292-322, has no analogue here — delete-by-design).

Inference is identity-only (no test-time symmetry ensembling), as in production today.

---

## 6. Search integration (the lens centerpiece)

### 6.1 Shape of the system

Greenfield crate `hexspan_native` (Rust owns search; Python/Torch owns the forward), modules:
`state.rs` (Py→Rust state intake), `encode.rs` (§2 featurizer, both entry points), `eval.rs`
(cache/dedup/payload/parse — mcts_eval.rs semantics), `tree.rs` (PUCT, prior-sorted lazy edges,
nucleus widening 0.95/96/2, FPU, virtual loss, subtree reuse via root advance, TSS injection +
leaf override + root guard), `mcts.rs` (lockstep `search` + `run_continuous` with PCR, policy-init,
flush decisions, per-class noise/FPU/temperature, on_move callback).

**Fresh implementation, not extraction — justification:** lock #10 mandates greenfield code; extracting
tree/scheduler into a shared crate would refactor the live dense_cnn lineage mid-main_4 for zero
search-economics payoff. Semantics are pinned instead by **golden tests**: the new crate's `mix_seed`
uses the identical mixer and stream constants (SEED_STREAM_* 0..5, dense_cnn/rust/src/mcts.rs:61-66)
and must reproduce the golden values in mcts.rs:2673-2686; the flush-decision table
(mcts.rs:2688-2707), completion rule (`completed ≥ target && in_flight == 0`, mcts.rs:280-282), and
nucleus-count properties are mirrored as new tests asserting the same constants. TSS is **linked, not
rewritten**: `#[path]`-include of `packages/hexo_models/rust/src/threats_shared.rs` (the cross-package
include has explicit precedent: hexo_models/rust/src/lib.rs:44-46 includes ../../../hexgnn with a
documented fragility caution; same caution applies and is accepted). `tss_enabled` is wired through
both schedulers exactly as dense_cnn does (mcts.rs:752, 851). One crop-era behavior is deleted rather
than ported: tactical-cell in-crop intersection — every tactical cell is always legal
(threats_shared.rs:34-38) and our vocabulary is the full legal set, so TSS injection never meets a
representational boundary. dense_cnn's `frozen_win_override` (a crop disease patch, main_4.toml C3)
has no reason to exist and is not ported.

### 6.2 Evaluator payload — `kind = "hexspan_csr_v1"`

Per flush of G unique cache-missed states, CSR flat-concat over T = Σ N_g nodes. Rust builds it with
the rayon-parallel encode + zero-copy buffer pattern proven in mcts_eval.rs:241-332.

| key | dtype | shape | bytes | notes |
|---|---|---|---|---|
| `kind` | str | — | — | fail-loud ABI tag |
| `inputs` | f16, zero-copy buffer-protocol object | `[T, 13]` row-major | 26·T | PlaneBuffer pattern (mcts_eval.rs:48-100): no GIL-thread memcpy |
| `coords` | i16 PyBytes | `[T, 2]` | 4·T | axial (q,r) per node, for on-GPU bias indexing |
| `nbr` | **u16** PyBytes | `[T, 6]` | 12·T | graph-LOCAL neighbor index per DIR order (§3.2); sentinel `0xFFFF`; guard: error if any N_g > 65,534 |
| `row_offsets` | i64 PyBytes | `[G+1]` | 8(G+1) | CSR node offsets; `row_offsets[G] == T` validated |
| `legal_counts` | i32 PyBytes | `[G]` | 4·G | L_g; legal prefix lengths |
| `shape` | tuple | `(G, T)` | — | cross-checks every buffer length |

**42 bytes/node.** Sizes: G=256 mid-game (N̄=900): **9.7MB** vs dense_cnn's 11.2MB planes + 1.6MB
legal flats = 12.8MB (−24%); early game (N̄=300): 3.2MB (−75%). No legal-index arrays exist at all
(legal-prefix property, §2.2). Return D2H stays small: ~0.8MB/flush.

**Response (unchanged contract family):**
`{ values_bytes: f32[G] (clamped [−1,1] — read_value contract, mcts_eval.rs:609-617),
priors_bytes: f32[ΣL_g] positional in node order 0..L_g per graph,
moves_left_bytes?: f32[G] (only when requested — §6.5) }`.
Rust zips priors with the per-graph sorted `Vec<PackedCoord>` it kept, validates
(finite/≥0/unique/mass>0), sorts descending, normalizes — the full finalize_model_priors discipline
(mcts_eval.rs:515-580) — then **truncates for storage** (§6.4).

### 6.3 Python evaluator: sort-and-pack into static shapes

This is where variable-N is reconciled with the kernel-shape economics (cuDNN re-autotune ≈ 925ms per
novel conv shape; compile graphs are per-shape; restnet's A7 bucketing is load-bearing — brief §4).
`HexspanInference.evaluate_payload`:

1. **One H2D copy per array** (frombuffer → cuda, non_blocking).
2. **Deterministic packing (CPU, numpy, ~tens of µs):** sort graphs descending by N_g (ties: arrival
   index). Greedily emit fixed-shape batches from the static shape table; a graph fits any shape with
   `S_c ≥ N_g`, so smaller graphs fill the tail rows of larger-shape batches; the final partial batch
   zero-pads its rows. Packing affects only performance, never values (rows are independent), but is
   deterministic for reproducible perf and tests.

   **Static shape table (7 shapes total — the whole compile/autotune surface):**

   | S_c (cell slots) | B (rows) | tokens/forward (B·S_c) | bias mask `B·4·S²·2B` (S = S_c+8) |
   |---|---|---|---|
   | 384 | 64 | 24,576 | 79 MB |
   | 512 | 48 | 24,576 | 104 MB |
   | 768 | 32 | 24,576 | 154 MB |
   | 1024 | 24 | 24,576 | 204 MB |
   | 1536 | 16 | 24,576 | 305 MB |
   | 2048 | 8 | 16,384 | 135 MB |
   | 3072 | 4 | 12,288 | 152 MB |

   Uniform ~24.5k cells/forward through the common range (uniform GPU work quantum); the two
   long-tail shapes trade rows for the quadratic mask. Worst transient mask ≈ 305MB (§10.4).
3. **Scatter CSR → padded:** `feat [B, S_c, 13]`, `coords [B, S_c, 2]`, `nbr [B, S_c, 6]`,
   `valid [B, S_c]` via one `index_copy_` per array with precomputed flat destinations
   (`row · S_c + local`). Neighbor globalization on GPU: `mask = nbr != 0xFFFF`,
   `safe = where(mask, nbr, 0) + row_base`.
4. **Forward per shape** (fp16 weights; compiled per shape after M5): policy logits `[B, S_c]`,
   value logits `[B, 65]` → decoded scalar, optional moves_left.
5. **Per-graph prefix softmax** over slots `[0, L_g)` using the segment scatter-softmax already proven
   in inference.py:385-408 (one max-reduce, one exp, one sum-scatter; no per-row Python).
6. **Single D2H sync:** `cat(values, priors, zero_mass_flag)` → one `.cpu()` (the proven single-sync
   trick, inference.py:403-406), reorder to original graph order, emit bytes.

**Why this owns the lens:** GPU shape stability is the *evaluator's* property, not the scheduler's.
A 54-leaf staggered flush, a 256-leaf continuous flush, and a 768-leaf lockstep round all decompose
into the same 7 shapes — schedulers can stagger freely with zero re-autotune/compile churn, and the
avg-batch-54 pathology class (brief §4) is absorbed at the packing layer.

### 6.4 Cache + prior truncation

`HashMap<StateHash, Arc<Evaluation>>`, FIFO-bounded, key = `hexo_utils::hash_state` (pure engine board
hash — encoder-independent, so cache behavior is identical to today's by construction). Default bound
262,144 entries (main_4 production value, dense_cnn_restnet_main_4.toml:211).

**Stored prior truncation (deviation §12.3):** after finalize, keep
`top-max(96, widening_max_children) ∪ tactical_cells(state)`; record `legal_action_count = L_g` (full).
*Tree-exactness proof:* widening materializes children in descending-prior order and never beyond
`max_children = 96` (lock #10's 0.95/96/2), so no entry past rank 96 is ever consumed — visit counts
are bit-identical (pinned by a property test over seeds, §9). The single semantic touch-point is
policy-init Init moves, which sample "from the raw root prior" (mcts.rs:744): support becomes
top-96 ∪ tactical (≥95% of mass; each dropped tail cell holds <0.01%). Declared; config
`eval_prior_keep = 0` restores full lists. Payoff: cache entry ~0.93KB vs ~7.4KB untruncated
(measured ~6.6KB/state today) → 262k entries ≈ **245MB instead of ~1.7GB**, and tree hidden-prior RAM
(4.2GB at 256 visits historically) drops ~9× at the node level.

### 6.5 Optional moves-left output

`request_moves_left: bool` on the session config adds `moves_left_bytes` (f32[G], decoded scalar in
[0, 512]) — zero cost when off (aux trunk skipped at search), forward-compatible with the owner's
moves-left-utility track (commit fe0e17c). Default off; not consumed by PUCT in v1.

### 6.6 Scheduler semantics (preserved exactly)

Both schedulers ship: lockstep `search` and `run_continuous` (per-slot ContinuousSlot machinery,
RootInit queue items, flush decisions, select↔eval overlap with the in_flight>0 completion guard —
mcts.rs:940-1005). PCR (`pcr_full_proportion`, fast visits, per-class noise/forced-k/FPU/temperature),
policy-init (truncated-exponential ply draw, temperature sampling), root noise
(`root_noise_exact` per-edge Dirichlet), root-FPU-zero-under-noise opt-out, temperature_by_ply,
forced-playout pruning of exported targets, deterministic `mix_seed(base, game_key, ply, stream)` —
all per dense_cnn behavior with golden parity (§6.1). Defaults mirror main_4 production: 512 visits,
c_puct 1.5, widening 0.95/96/2, vloss 1.0, fpu 0.20, active_root_limit 256.

---

## 7. Data pipeline

### 7.1 Shard schema `hexspan_compact_v1` (one .npz + json sidecar per game)

Same compact-facts concept (raw facts, representation-agnostic — brief §4), with three deletions vs
the dense_cnn_restnet schema (compact_io.py:185-222):

- **dropped `center_q/r`** — no crop, no center;
- **dropped `own_hot/opp_hot/last_hot` coordinate lists** — derived facts; the Rust facts-featurizer
  recomputes windows from stones at expand time (removes both redundancy and the ≥3-vs-≥4 threshold
  coupling);
- **`phase` as u8 enum {0,1,2}** instead of object-dtype strings (no pickle, faster).

Kept per row: `turn_index i32, current_player u8, value f32, moves_left f32 (−1 = masked),
first_present u8 + first_q/r i16, stvalue [N,3] f32 + mask`; CSR var-len: `stones_qr i16 +
stones_owner u8, hist_qr i16 + hist_owner u8 + hist_idx i32, legal_ids u32, pol_act u32 + pol_w f32,
opp_act u32 + opp_w f32`. Sidecar: outcome, lengths, policy-surprise stats (same fields as today).
`.hxr` game records unchanged (shared infra).

### 7.2 Expand-time featurization

One PyO3 call per shard: `hexspan_expand_rows(facts, symmetries) →` the same CSR arrays as the search
payload (§6.2) **plus** per-row target CSR (`pol_slot u32, pol_w f32` mapped to legal-prefix slots
after D6; same for opp with projection + dropped-mass count) and scalar targets. Runs in the trainer's
worker processes; Rust does the per-row work (rayon inside the call), Python only collates.

### 7.3 Collation & trainer deltas

Training pads per batch to `max N_g` rounded up to 64 (no fixed shape table needed — training is AMP
eager; matmul/sdpa tolerate fresh shapes without autotune cliffs). Batch tensors: feat f32, nbr i64,
valid bool, coords i32, dense policy/opp rows scattered from CSR, masks. Optimizer/regime identical to
restnet (harvest lock #11): AdamW lr 1e-3, wd 1e-4 on matrix weights only, AMP, grad-clip 1.0,
batch 32, per-row D6. Replay identical: mtime-ordered KataGo tapered window (keep 300k, taper 0.65),
policy-surprise row materialization at finalize (replay.py:178-268). Plugin: entry-point group
`hexo_train.models`, `hexspan = hexspan.plugin`, implementing build_model /
training_component_overrides / generate_selfplay / select_training_samples / train_passes /
evaluate_epoch against the shared pipeline (hexo_train/registry.py:38-58; pipeline/config/diagnostics
shared, unchanged).

---

## 8. Bootstrap

1. **BC prefit** from `timmyburn/hexo-bootstrap-corpus` (6,902 games ≈ 431k positions): replay raw
   move lists through hexo_engine (scripts/bootstrap_dense_cnn_restnet_hf.py pattern), emit
   `hexspan_compact_v1` shards with one-hot played-move policy, hard-z values, moves_left from
   remaining game length, no stv (masked). Prefit batch 48–64 (WSL note: dense prefit batch 64 fits,
   128 OOMs; ours is lighter mid-game but variable — start 48). Gate: held-out top-1 legal accuracy ≥
   the dense_cnn BC reference at equal passes; value sign-accuracy on decided positions.
2. **Distillation from existing self-play shards (recommended, cheap):** the restnet compact shards
   are raw facts (brief §4 — two lineages already expand the same shards differently). A read-adapter
   maps their schema (ignore `center_*`/hot lists; parse phase strings) into the same expand path —
   millions of 512-visit MCTS policy/value/stv/moves-left rows from main_4's window for free, before
   the first hexspan self-play epoch. Zero new mechanism: it is the §7.2 path behind a different reader.
3. Order: BC prefit → shard distill → self-play RL. (GPU scheduling out of scope.)

---

## 9. Code architecture

```
packages/hexspan/
  pyproject.toml                      # maturin build; entry point hexo_train.models: hexspan
  python/hexspan/
    constants.py geometry.py          # axial D6 transforms (fresh file, same math as d6.py)
    architecture.py                   # HexspanNetwork; sdpa + materialized oracle; masked BN
    inference.py                      # HexspanInference: payload ABI, packing, fp16 gate, compile
    samples.py compact_io.py replay.py losses.py trainer.py selfplay.py
    plugin.py config.py rust_bridge.py checkpoints.py
  rust/                               # crate hexspan_native (its own cdylib — does NOT join
    src/lib.rs                        #   hexo_models._rust, so main_4's module is never rebuilt)
    src/{state,encode,eval,tree,mcts}.rs
    src/threats_shared -> #[path] include of packages/hexo_models/rust/src/threats_shared.rs
```

Links against (and only against): **hexo_engine** (Rust dep: state, legal, windows, pack_coord),
**hexo_utils** (hash_state), **hexo_train** (Python pipeline), **threats_shared** (file include).
No imports from dense_cnn/restnet/hexgt/hexgnn code anywhere (lock #10); they are referenced only in
tests' golden constants and docs.

**Test strategy (the contracts that must not drift):**

1. *Featurizer parity:* random engine games → `encode_state(s)` vs `encode_facts(facts(s))` byte-equal
   CSR (the keystone; replaces dense_cnn's Rust-vs-Python alignment burden).
2. *Halo theorem:* support construction vs brute-force d=9 ring on 1k random states.
3. *Attention oracle:* sdpa vs materialized ≤ 1e-3 rel on random graphs, both dtypes.
4. *Padding independence:* perturb padding rows/cols → valid outputs bit-stable (the invariant that
   licenses every padding shortcut).
5. *Masked-BN:* equals gather-BN-scatter reference; running stats under variable N.
6. *D6 round-trip:* transform→inverse on facts is identity (geometry.py basis tests).
7. *Evaluator ABI:* malformed-bytes table mirrored from mcts_eval.rs validations (byte counts,
   offsets monotone, NaN/negative/duplicate priors, zero-mass rows) — every row fail-loud.
8. *Seed goldens:* mix_seed values equal dense_cnn's golden test constants; flush-decision table;
   completion rule.
9. *Truncation equivalence:* search visit counts identical with `eval_prior_keep` ∈ {0, 96+tactical}
   across seeds at widening 0.95/96/2.
10. *E2E:* self-play smoke vs a stub evaluator (CPU) + a short GPU epoch through the shared pipeline.

Build: maturin develop in the WSL venv (tests authoritative there — known environment fact);
`cargo test` covers pure-Rust modules (tree/scheduler logic) without the `python` feature, mirroring
the existing limitation that PyO3-linked tests run via pytest e2e.

---

## 10. Perf budget

### 10.1 Parameters — total **1,232,443 (~1.23M)**; fp16 2.46MB, fp32 4.93MB

| component | params |
|---|---|
| stem DirConv7 13→96 (no bias) + BN | 8,928 |
| 6 × C block (2 × [7·96² conv + BN]) | 776,448 |
| 3 × A block (QKV+O 37,248; MLP 37,152; 2 LN 384) | 224,352 |
| rel-pos bias table 301×4 (shared) + 8 tokens ×96 | 1,972 |
| policy head + opp head (7·96²+96 conv, 96→1) ×2 | 129,410 |
| value MLP (192→128→65) | 33,089 |
| aux trunk + 4 tops (192→128; 4×128→65) | 58,244 |

Search-time forward touches 1.109M of these (opp head + aux skipped unless moves_left requested).

### 10.2 FLOPs per eval (MACs; ×2 for FLOPs). `S = N + 8`

`MACs(N) ≈ 847,584·N (stem+6C+policy head) + 221,184·S (A proj+MLP) + 576·S² (attention pairs) + 33k (value)`

| position | N | hexspan GF | restnet GF (fixed) | ratio |
|---|---|---|---|---|
| opening | 300 | 0.75 | 5.8 | **0.13×** |
| mid-game median | 900 | 2.87 | 5.8 | **0.50×** |
| long | 1,500 | 5.83 | 5.8 | 1.0× |
| extreme tail | 3,000 | 16.8 | 5.8 (and *blind* — the crop) | 2.9× |

restnet reference ≈ 2.89 GMACs/eval: 6R = 1.673G (2×9-tap×96²×1681), 2T KV-gathered = 1.062G
(proj/MLP 124M + pairs 2·1681·1261·96), stem 19M, policy head 140M. The brief's "at N≈1000 an A layer
≈ a C block" reproduces here: A-block/node at N=1000 ≈ 267k MACs vs C-block 129k (≈2× incl. MLP).

### 10.3 Throughput model at 512 visits

Anchors (measured, restnet): 4,561 evals/s isolated forward at 5.8 GF/eval = **26.4 TFLOP/s
effective** with fp16+KV-gather+compile; eager 3,589 (20.8 TF/s). hexspan's mix (gathers + masked
sdpa + GEMMs, no cuDNN convs) is assumed to land at **16–20 TF/s effective** (deliberately
conservative: −25–40% vs restnet's conv-heavy profile), and packing adds ~1.25× effective FLOPs
(row padding within sorted batches + final partial batches).

- Mid-game (2.87 GF × 1.25): **4,500–5,600 evals/s** (vs restnet 4,561 — parity to +23%).
- Opening probe regime (0.75–1.5 GF): **2–4× restnet's rate**; the 2048-opening-position calibration
  probe should clear restnet's 8.79 pos/s decisively.
- Game-average at ~411 unique forwards/decision (measured; cache hit 1.2% at 512 visits — unchanged
  here since the cache key and tree shape are preserved): **+20–60% positions/s**, with the floor case
  (utilization lands at 13 TF/s) being parity. The lineage's case does not rest on the speedup — it
  rests on paying *position-sized* cost with *zero coverage loss*; the speedup is the lens's dividend.

Known #1 inefficiency, priced: the per-graph bias mask. At shape (24, 1024): build-write 204MB +
3 sdpa reads ≈ 0.8GB traffic ≈ 1.6ms vs ≈3.6ms compute — **~25–30% of A-layer time, ~10–15% of the
forward**. Bounded, accepted for v1 (still beats the dense alternative), escape hatch in §10.5.

Rust featurization: target ≤150µs/leaf at N=900 single-thread (support hash-build ~5k probes, 6×N
neighbor probes, window scan); rayon across the flush (mcts_eval.rs:248-251 pattern) → 256 leaves ≈
2.4ms on 16 threads ≈ **<5% of wall** (dense encode measured ≈7%).

### 10.4 VRAM

- **Inference:** weights 2.5MB; per-forward transients: x 4.7MB (S_c=1024 shape), conv gather
  `[B,S_c,7,96]` f16 ≈ 33MB, qkv ≈ 14MB, bias mask ≤ **305MB worst shape** (table §6.3); compile
  (reduce-overhead) pools shared across the 7 graphs (cudagraph-tree pooling) — budget **≤1.0GB**
  total inference arena.
- **Training (batch 32, N_pad ≈ 1100):** bias mask 314MB (single shared buffer per step), saved conv
  gathers 12 × 47MB ≈ 568MB (contingency if tight: recompute-gather-in-backward, listed not built),
  other activations ≈ 300MB, params+grads+Adam ≈ 60MB → **≈1.5–2.5GB peak** — comfortable next to the
  inference arena on 12GB; batch 32 stands (restnet's batch-64 OOM was at 1681 fixed tokens; we do not
  raise batch in v1).
- **Host:** eval cache 245MB at 262k entries (truncated, §6.4); tree hidden priors ~9× smaller than
  the historical 4.2GB figure.

### 10.5 fp16 / compile / TRT story

- **fp16 weights** day one, behind restnet's exact adoption gate (argmax match ≥0.90, decoded value
  err ≤0.05 on real positions — inference.py:455-467). f16 is already the transport dtype.
- **torch.compile** (`mode="reduce-overhead"`, fullgraph) per static shape — 7 graphs;
  `cache_size_limit` bumped to 4× shape count (the restnet gotcha, fixed the same way). Our bias is
  built from runtime tensors + a Parameter (no Python-cached constants), so the freeze-bias workaround
  class is structurally unnecessary; the only compile risk is the gather/scatter mix (M5 gates it).
  Per-shape persistent input staging buffers satisfy CUDA-graph static-address requirements; warmup
  walks all 7 shapes at init (the _warm_up_cuda pattern).
- **TRT: explicitly not in v1.** The measured 2.4–2.7× was on the *static-shape dense conv* model at
  bs128/256; hexspan's data-dependent gathers + per-graph additive-bias sdpa make ONNX/TRT export a
  poor bet. Compile+CUDA-graphs captures the launch-overhead share of that win.
- **FlexAttention upgrade path (v2 lever, not v1):** score_mod computing the §3.5 index from coords
  and reading the 2.4KB table inline eliminates the bias mask's HBM traffic *and* enables
  block-diagonal jagged batching (no padding at all). Parked behind the repo's lever discipline:
  adopt only on oracle-equivalence + ≥10% measured probe win (the KV-gather precedent).

---

## 11. Milestones (each gate blocks the next; no GPU-scheduling decisions anywhere)

| # | deliverable | acceptance gates |
|---|---|---|
| M0 | `encode.rs` (state + facts entries), geometry | parity test 1–2 (§9) green; bench ≤150µs/leaf @N=900 1-thread; ply-0/N=7 case |
| M1 | architecture.py (oracle + sdpa, masked BN, tokens, heads) | tests 3–5 green; param count == 1,232,443; CPU forward determinism |
| M2 | trainer + shard schema + expand + BC prefit | BC gates (§8.1); D6 round-trip; loss components finite & decreasing; held-out top-1 ≥ dense_cnn BC reference |
| M3 | evaluator boundary (payload ABI, packing, prefix softmax, fp16 gate) | test 7; padding-independence on packed batches; fp16 gate passes; **eager bench ≥2,500 evals/s** at shape (32,768) on synthetic mid-game states |
| M4 | tree + lockstep + continuous + PCR/policy-init/TSS + cache/truncation | tests 8–9; e2e smoke (10); TSS parity vs threats_shared diagnostics; short GPU epoch produces valid shards through the shared pipeline |
| M5 | compile adoption | per-shape fullgraph; output-equivalence vs eager (fp16 tolerance); adopt only at ≥+10% isolated-forward win, else park (lever discipline) |
| M6 | bootstrap ladder: BC → restnet-shard distill → first RL epochs | anchored eval wiring (ckpt-anchor ladder pattern); self-play probe ≥ restnet's pos/s on the same machine; padding-waste + bucket-histogram + opp-spill counters in epoch diagnostics from the first epoch |

---

## 12. Envelope deviations (all from §3 defaults; **no §2 contradictions found**)

1. **Rel-pos bias table shared across the 3 A layers** (default plausibly per-layer). Why: the bias is
   per-graph (coords), so it must be gathered every forward; sharing makes that 1 gather instead of 3
   (~550MB traffic saved per mid-shape forward) and 1 resident mask. Layers differentiate via content
   (Q·K). Param delta −2.4k. Risk: low; revisit only if A-layer probes show geometric specialization.
2. **17×17 dense offset indexing (289 slots, 72 unreachable)** instead of an exact 217-entry disk
   bijection: branch-free `(Δq+8)·17+(Δr+8)` on GPU; +288 dead params. Table = 301 rows/head total.
3. **Stored-prior truncation** to top-max(96, widening_max_children) ∪ tactical (§6.4): bit-exact for
   tree/widening (proof + property test); declared semantic touch on policy-init sampling support
   (≥95% mass retained); `eval_prior_keep = 0` off-switch. ~7× cache/tree RAM saving.
4. **HOT_COUNT_MIN = 3 kept** per the brief's default, with the explicit note that dense_cnn
   production used ≥4 (encoding.rs:238); config knob, flip without schema change (hot is derived at
   expand, not stored — §7.1).
5. **Tokens prepended at slots [0,8) from A1 onward**; cells at [8, 8+N_pad); legal prefix offset +8.
   Pure layout pin (compile-friendly static slices; token keys double as the never-masked rows that
   eliminate the all-masked-softmax NaN class).
6. **PAD_BIAS = −30000.0 finite** (vs restnet's −1e9-saturates-to-−inf pattern) — same masking effect,
   no inf arithmetic in fused kernels.
7. **u16 neighbor indices** with a 65,534-node fail-loud guard (observed max ~3k; 64× headroom);
   halves nbr payload vs i32.
8. **opp_policy target projected** onto the current support's legal slots + renormalized +
   `allow_zero_rows` + dropped-mass counter (restnet trains it unmasked over the whole crop; natively
   there is no "whole crop"). Spill is structural for next-decision targets near fresh stones;
   measured and reported, not silently absorbed.
9. **Optional `moves_left` in the search response** (off by default) — forward-compatibility with the
   owner's moves-left-utility track at zero cost when off.
10. **Schema deltas** (§7.1): drop center/hot columns, phase as u8 — all derivable facts; enables the
    restnet-shard distillation adapter to be read-only.
11. **No TRT in v1; compile+CUDA-graphs instead** (§10.5) — the TRT datum was measured on a
    static-shape model and does not transfer; declared so nobody chases it early.

— end —
