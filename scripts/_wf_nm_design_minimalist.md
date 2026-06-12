# hexfield — a stone-anchored, hex-native lineage (minimalist design)

Design competition entry. Lens: **radical simplicity** — fewest mechanisms, fewest knobs, shortest
path to a trainable, debuggable model. Every section below is implementable as written; where the
brief offered options, the simplest one that preserves the locked envelope was chosen, and every
deviation from the brief's §3 defaults is listed in §12.

Conventions used throughout: `N` = number of board nodes in one position's support set; `S = N + 8`
= attention sequence length (8 summary tokens prepended); `C = 96` channels; `F = 13` node features;
`B` = positions per batch; `T` = total node slots in a batch (token budget). Action id = packed u32
`((q + 2^15) << 16) | (r + 2^15)` (universal; `d6.py:41-59`). All `file:line` cites are repo paths.

---

## 1. Identity

**hexfield**: the model's domain is the *field* the stones induce — the engine-true support set
(stones ∪ full legal set ∪ 1-ring halo), realized as a variable-length set of hex nodes with
direction-typed 7-tap convs for local reasoning and restnet-style global attention with 8
carried-through summary tokens. The design thesis is *one of each*: **one geometric law** builds the
support set (and yields the distance feature as a free byproduct); **one norm** (LayerNorm)
everywhere; **one relative-position bias table** per attention layer with a closed-form index
function; **one batch layout** (padded, bucketed) shared by convs and attention; **one memory knob**
(the token budget) that bounds worst-case VRAM; and **one wire format back to Rust** — the reply
half of the evaluator ABI is byte-identical to dense_cnn's proven `values_bytes`/`priors_bytes`
contract, so the entire PUCT tree consumes hexfield evaluations exactly as it consumes dense_cnn's.
The crop is gone by construction: every engine-legal cell carries a policy logit, so coverage loss —
the root cause of the main_3 collapse — is structurally impossible. An engineer can hold the whole
model in their head: support-BFS → 13 features → C C C A C C A C A → five heads.

---

## 2. Input representation

### 2.1 Support-set construction (the one geometric law)

Ground truth is the engine, never re-derived geometry:

1. `stones` = `state.board().occupied_cells()`; `legal` = `state.write_legal_action_ids(...)`
   (engine truth; `hexo_engine/rust/src/state.rs:216-254`).
2. `core = stones ∪ legal` (as coordinate set).
3. `halo = { c + d : c ∈ core, d ∈ D } \ core` where `D` is the 6 axial unit directions (§2.3).
   Halo cells carry features but **never** logits (owner-locked).
4. `support = core ∪ halo`. **Canonical node order: sort by `(q, r)` ascending** (i16 lexicographic).
   Deterministic, cheap (≤ ~3k entries), and makes Rust↔Python parity tests exact-match.

Equivalent geometric fact (used for the distance feature, not for construction): since legality is
"empty ∧ hex-dist ≤ 8 of any stone" (`LEGAL_RADIUS=8`), `core` = all cells within distance 8 of a
stone and `halo` = exactly the distance-9 shell, so `support` = a multi-source BFS of depth 9 from
the stones. One BFS pass over the support (O(N), sources = stone nodes) therefore yields
`dist_to_stone` for every node *and* verifies the halo. Shortest paths stay inside the support
(every intermediate cell of a shortest path to a stone is closer to that stone), so BFS distance
= true hex distance `max(|dq|,|dr|,|dq+dr|)`.

Scale anchors: 1 stone → 217-cell radius-8 disk + 54-cell halo ring = 271 nodes; mid-game ≈
600–1500; long spread games ≈ 3k (brief §4). No upper cap exists anywhere (a cap would be a crop).

**Edge cases.**
- *Ply 0 (empty board)*: `stones = ∅`, engine legal = `{(0,0)}` (forced origin). Support =
  origin + its 6 neighbors = 7 nodes. `dist_to_stone` has no sources: defined as 0.0 for all nodes.
  The move is forced, so the network output is irrelevant here; the rule exists only so the forward
  is total.
- *Terminal states*: never evaluated by search (the tree backs up engine outcomes), but the payload
  path tolerates a zero-legal row (returns a value, zero priors) — same semantics as
  `finalize_model_priors`'s terminal-row branch (`dense_cnn/rust/src/mcts_eval.rs:522-530`).
- *Disconnected stone clusters*: support is a disjoint union; convs stay local per component,
  attention (with the far-distance bias bucket) spans components. This is exactly the out-of-rim
  case the crop could not represent.

### 2.2 Node feature table (F = 13, exact port of the trusted planes)

Same indices as dense_cnn's plane constants (`dense_cnn_restnet/constants.py:29-41`) so every
feature has a battle-tested reference semantic; only index 11 is redefined (crop-center distance →
distance-to-nearest-stone, per brief §3). All features f32 at train time, f16 on the wire (§6).

| idx | name            | value (per node v) |
|-----|-----------------|--------------------|
| 0   | own_stone       | 1.0 if stone owned by side-to-move |
| 1   | opp_stone       | 1.0 if opponent stone |
| 2   | empty           | 1.0 if not a stone (legal ∪ halo) |
| 3   | legal           | 1.0 if v ∈ engine legal set |
| 4   | phase_second    | constant 1.0 on all nodes iff phase == SecondStone |
| 5   | first_stone     | 1.0 at the turn's first placement cell iff phase == SecondStone (that cell is a stone, hence in support) |
| 6   | player_colour   | constant 1.0 on all nodes iff side-to-move == player0 |
| 7   | own_recency     | max over own placements at v of `1/(1 + latest_idx − placement_idx)`, `latest_idx = placements_made` (`dense_cnn/rust/src/encoding.rs:182-196`) |
| 8   | opp_recency     | same for opponent placements |
| 9   | opp_hot         | 1.0 if v is an EMPTY cell of a single-colour opponent window with count ≥ 4; gated `placements_made ≥ 7` (`encoding.rs:226-266`) |
| 10  | own_hot         | same for own windows |
| 11  | dist_to_stone   | `min_hex_dist(v, stones) / 8.0`; stones → 0.0; legal ∈ (0,1]; halo = 1.125 exactly; no stones → 0.0 |
| 12  | opp_last_turn   | 1.0 at the cells of the opponent's most recent full turn (`encoding.rs:268+`) |

Notes:
- The hot features keep dense_cnn's **count ≥ 4** definition deliberately: it is the *same*
  threshold as a TSS "threat" (`hexo_models/rust/src/threats_shared.rs:21-23`), so the repo has one
  concept of "threat window" feeding both features and search. (The brief's "≥count-3" wording is
  refined back to the trusted semantics — §12.)
- No separate "halo" flag: halo ≡ `empty=1 ∧ legal=0`. No new feature invented.
- All values are exactly representable in f16 (binary flags, k/8 grid, 1/(1+age) ≤ 1) — the f16
  transport is loss-free for search, matching dense_cnn's gate rationale (`mcts_eval.rs:270-272`).

---

## 3. Trunk

Interleave (brief §3 default, adopted): **`C C C A C C A C A`** — 9 layers, 6 conv residual blocks,
3 attention blocks. Rationale per layer: stem + C1–C3 give local receptive radius 7 (stem 1 + 2 per
block) ≥ the 5-step span of a 6-window before any global mixing; A4 introduces the tokens; C5–C6
re-localize with token-informed features; A7 gives the token hub its second round (bidirectional
aggregation needs ≥ 2); C8 sharpens locally; A9 ends the trunk so cells *and* tokens are maximally
fresh for the heads.

State threading: trunk state is `(x, t)` — `x` cell features `(B, S_pad−8 ≡ Npad, C)`, `t` token
features `(B, 8, C)`. C blocks update `x` only; A blocks update the joint sequence
`seq = concat([t, x], dim=1)` of length `S_pad = Npad + 8` and split back. Tokens are initialized
ONCE from a learned `(8, C)` parameter at A4 and **carried through** A7/A9 (never re-initialized).

### 3.1 Stem

`x = ReLU(LN(HexNodeConv_{13→96}(features)))` — EmbedNet semantics (conv → norm → ReLU,
`dense_cnn_restnet/architecture.py:48`), with the one norm (§3.2) and the one conv primitive.

### 3.2 HexNodeConv (the direction-typed 7-tap primitive)

Weight `W ∈ (7, C_in, C_out)`, bias `(C_out,)`. Tap 0 = center; taps 1–6 = the fixed direction
order `D = [(1,0), (0,1), (−1,1), (−1,0), (0,−1), (1,−1)]` (rotate60-cyclic: `rot60(D[i]) =
D[(i+1) mod 6]`; reflect maps `D[i] → D[5−i]` — used only by tests, see §5).

Forward (padded layout): `g = gather(x_flat, nbr_idx)` → `(B, Npad, 7, C_in)` where `nbr_idx`
`(B, Npad, 7)` holds tap 0 = self and taps 1–6 = global flat indices of `v + D[d]`, or the index of
an appended all-zeros row when the neighbor is outside the support (= conv zero-padding semantics,
brief §3). Then one GEMM: `y = reshape(g, (B, Npad, 7·C_in)) @ reshape(W, (7·C_in, C_out)) + b`.
This is mathematically the dense_cnn hex conv family — one weight matrix per relative direction,
shared everywhere (owner-locked §2.3) — expressed as a gather + single GEMM. `nbr_idx` is built once
per batch and shared by all 15 conv applications (stem + 12 trunk + 2 head convs).

Pad rows compute garbage but never contaminate real rows: neighbor indices only ever point at real
nodes or the zero row, attention masks pad keys (§3.4), LayerNorm is per-node, and every loss/output
is masked to valid nodes. No re-zeroing pass needed (restnet's proven masking discipline,
`architecture.py:88-97`).

### 3.3 Conv residual block (×6)

Post-activation residual, dense_cnn family (`architecture.py:237-259`), with LN in place of BN:

```
y = ReLU(LN1(Conv1(x)))          Conv1, Conv2: HexNodeConv C→C, bias=True
y = LN2(Conv2(y))
x = ReLU(x + y)
```

**Norm choice = LayerNorm, and only LayerNorm, everywhere** (stem, conv blocks, attention blocks,
final norm). Justification for overriding the §3 default (BatchNorm with LN fallback): (a) on
variable-N node sets, batch statistics depend on batch composition (a 7-node ply-0 row and a 3k-node
marathon row in one batch), and padded rows must be excluded from the statistics — masked BN is
implementable but stateful and fiddly; (b) LN has no running stats, no train/eval divergence, no
conv-BN folding machinery (deletes the whole `optimized_*_for_inference` fusion class of code,
`architecture.py:888-958`); (c) the attention blocks need LN anyway — one norm = one mechanism. The
empirical tripwire is the M3 BC prefit (§11): if LN convs clearly undertrain vs the restnet BC
reference, the documented contingency is masked BatchNorm1d over valid nodes. There is no runtime
knob — the knob is deleted.

### 3.4 Attention block (×3)

Pre-norm restnet transformer block, exact semantics of `architecture.py:576-620`:

```
seq = seq + Attn(LN1(seq))        # RelPos MHSA, 4 heads, head_dim 24, scale 1/sqrt(24)
seq = seq + MLP(LN2(seq))         # Linear C→2C, GELU, Linear 2C→C   (mlp_ratio 2)
```

Q/K/V/O are `Linear(C→C, bias=True)`. Two numerically identical attention impls share parameters:
`sdpa` (production: `F.scaled_dot_product_attention` with the bias as additive `attn_mask`) and
`materialized` (the correctness oracle) — restnet's proven dual-impl pattern
(`architecture.py:262-281`).

**Relative-position bias** — one learned table per A block, shape **(229, 4 heads)**:

| rows    | meaning |
|---------|---------|
| 0–216   | exact axial offset `(dq, dr)`, hex-dist ≤ 8 (217 cells of the radius-8 disk) |
| 217–224 | ring buckets, hex-dist 9–16 (one row per distance) |
| 225     | far bucket, hex-dist ≥ 17 |
| 226     | query = cell, key = token |
| 227     | query = token, key = cell |
| 228     | token ↔ token |

Index function for a (query i, key j) pair, with `Δ = (q_j − q_i, r_j − r_i)` and
`d = max(|dq|, |dr|, |dq+dr|)`:

```
idx(i, j) = EXACT_LUT[(dq+8)*17 + (dr+8)]   if both cells, d ≤ 8     (∈ [0, 216])
          = 217 + (d − 9)                   if both cells, 9 ≤ d ≤ 16
          = 225                             if both cells, d ≥ 17
          = 226 / 227 / 228                 per the token cases above
```

`EXACT_LUT` enumerates `(dq, dr) ∈ [−8,8]²` row-major, assigning sequential ids where d ≤ 8 —
fixed at import, Python-only (Rust never builds bias). Tokens sit at sequence positions 0–7 and
have **no board position**; their relation to all cells is the per-head scalars in rows 226–228
(owner-locked §2.6, refined to 3 rows by brief §3). Pad cells use coords (0,0); their KEY columns
get an additive −1e9 (restnet's `_DISK_MASK_VALUE` discipline, `architecture.py:97`). Token keys are
**never** masked, so every attention row — including pad-row queries — has ≥ 8 live keys; a fully
masked row is structurally impossible, which closes restnet's documented fp16 −inf saturation hazard
(`architecture.py:88-96`) by construction rather than by argument.

The integer pair-index tensor `(B, S_pad, S_pad)` is a pure function of node coords: **computed once
per batch and reused by all 3 A blocks** (each block gathers its own table through the shared
index). Per-sample bias is what the unbounded board costs; §10 budgets it and the token budget
bounds it.

Bias table init = zeros; `trunc_normal_(std=0.02)` on Linears; LN weight 1 / bias 0
(`architecture.py:876-886`). Token init vectors: trunc_normal 0.02. Conv weights: PyTorch conv
default (kaiming-uniform with fan_in = 7·C_in).

### 3.5 Final norm and head taps

After A9: one shared `LN_final` over the full sequence (pre-norm trunks need a terminal norm).
Then: cells `x_out = LN_final(seq)[:, 8:, :]` feed the policy heads; tokens
`t_out = LN_final(seq)[:, 0:8, :]` feed value/aux heads. Token assignment: **T0–T1 → main value,
T2–T3 → STV + moves-left aux, T4–T7 uncommitted hub capacity** (brief §3 default, adopted).

---

## 4. Heads & losses

All target semantics are exact ports of the verified restnet constructions, re-keyed from crop
flats to support nodes.

| head | tap | architecture | output shape | target | mask | weight |
|------|-----|--------------|--------------|--------|------|--------|
| policy | cells | HexNodeConv(C→C) → ReLU → Linear(C→1) | (B, Npad) | MCTS visit weights, action-id → node | legal nodes only | 1.0 |
| opp_policy | cells | same shape, separate params | (B, Npad) | next opponent decision's visit policy (`samples.py:330-354`) | valid (non-pad) nodes; `allow_zero_rows` | 0.25 |
| value | T0,T1 | concat(192) → Linear(192→96) → ReLU → Linear(96→65) | (B, 65) | hard z ∈ {−1,+1} (0 only truncation) → adjacent-bin soft target (`losses.py:33-53`) | — | 1.0 |
| stvalue_{2,6,16} | T2,T3 | shared concat(192) → Linear(192→96) → ReLU; per-horizon Linear(96→65) | (B, 65) ×3 | even-offset EMA of future root values, decay (m−1)/(m+1), side-to-move perspective (`samples.py:357-402`) | per-row mask (horizon absent) | 0.1 each |
| moves_left | T2,T3 (same shared reduction) | Linear(96→65) | (B, 65) | `clamp(remaining_decisions, 0, 512)/512 → [−1,1]` → 65-bin (`constants.py:27`, `samples.py:277-281`) | per-row mask (−1 sentinel = truncated/absent) | 0.1 |

- **Pathway separation (heads_v3 lesson, natively):** main value reads tokens {0,1} through its own
  private MLP; the non-stationary aux targets read tokens {2,3} through a separate shared reduction.
  Separation is by *token* and by *readout* — stronger than restnet's reduction split
  (`architecture.py:14-20`) with zero extra mechanism.
- **No spatial ownership/win-window head** (owner skipped it).
- Policy CE = masked soft cross-entropy with strict target validation, semantics of
  `losses.py:56-97`: legal-masked softmax support, fp32 upcast before the −1e9 fill, target mass
  outside mask → hard error. The MCTS target is supported on legal nodes **by construction** (the
  vocabulary is the engine legal set — zero coverage loss; an assert enforces zero dropped mass).
- opp_policy is intentionally NOT legal-masked (next decision is a different phase/position;
  `losses.py:148-153`); pad nodes are excluded via the valid mask. Target mass on cells outside the
  *current* support (possible: the next position's legality extends beyond dist 9) is dropped and
  renormalized — same boundary semantics as dense's out-of-crop skip, now telemetered as
  `opp_mass_dropped` (expected tiny; aux-only head).
- Value bins: 65 points `linspace(−1, 1)`; decode = softmax-expectation, clamped to [−1,1]
  (`losses.py:20-31`, `inference.py:366-372`). Hard-z (owner-locked §2.9); no soft-z blend in v1.
- Total loss = `1.0·policy + 1.0·value + 0.25·opp + 0.1·Σstv + 0.1·moves_left` (brief §4 production
  weights; the model1_loss surface `losses.py:134-176` is the semantic reference).

---

## 5. Symmetry — D6 by training-time augmentation

Owner-locked: augmentation, not architectural invariance. Mechanics:

- 12 transforms, index 0–11: 0–5 rotations, 6–11 reflect-then-rotate; `rot60(q,r) = (−r, q+r)`,
  `reflect(q,r) = (q, −q−r)` (`d6.py:129-136`). **Transform center = (0,0)** — the forced origin is
  the canonical fixed point; with no crop there is no per-sample center (one less moving part vs
  dense's crop-center transforms).
- Per training row, one symmetry is drawn (pipeline-supplied, as today) and applied to **all stored
  coordinate facts** before featurization: stones, placement history, first_stone, hot cells,
  last-turn cells, legal action ids, policy/opp-policy action ids.
- The support set is then *built from the transformed facts*. Construction commutes with D6
  (legality and adjacency are hex-distance-based, D6-invariant), so the augmented row is a valid
  position presented in a rotated frame.
- **Nothing else transforms.** Conv weights are not permuted (the data rotates, exactly as rotating
  an image under a square CNN); the bias index function reads transformed coords; `EXACT_LUT` is
  fixed. D6 maps the offset set and distance rings onto themselves, so every index stays in-range.
- Inference always runs identity (no test-time augmentation), matching production dense_cnn.

Tests (§9): (a) coordinate-level — support/feature/target construction commutes with each of the 12
transforms up to the canonical (q,r) re-sort (a permutation check); (b) direction algebra —
`rot60(D[i]) = D[(i+1)%6]`, `reflect(D[i]) = D[5−i]`.

---

## 6. Search integration

### 6.1 Decision: fresh implementation, one hosting crate

The search is **written from scratch** in hexfield's own Rust tree (owner-locked §2.10), hosted by
the existing one-crate pattern: `packages/hexo_models/rust/src/lib.rs` gains a fourth `#[path]`
include + submodule registration (`hexo_models._rust.hexfield`), exactly as hexgnn is hosted from
outside the package directory today (`hexo_models/rust/src/lib.rs:36-46`). This is a ~12-line diff
to one file, reuses the canonical rebuild script, and — decisively — gives hexfield
`crate::threats_shared` (it is `pub(crate)`; linking against it *requires* same-crate compilation,
which is precisely how dense_cnn and hexgt share it, `threats_shared.rs:5-9`).

**Not chosen:** extracting dense_cnn's `mcts.rs`/`mcts_tree.rs` into shared generics. That would
refactor the live main_4 lineage's search mid-run for zero v1 benefit. Extraction remains possible
later, after hexfield's rewrite is proven equivalent by the differential test below.

### 6.2 Preserved search semantics (the contract list)

From-scratch `tree.rs`/`search.rs` must reproduce, with the dense_cnn implementation as the
semantic reference:

- Batched PUCT with virtual loss; prior-sorted lazy edge materialization (priors arrive descending
  — the sort in finalize is load-bearing); nucleus widening `policy_mass 0.95 / max_children 96 /
  min_children 2`; FPU (`value_or_fpu`, separate `root_fpu_reduction`,
  `root_fpu_zero_under_noise` for Full-class roots only — `mcts_tree.rs:74-76,241-248,621-622`,
  `mcts.rs:182-185`); Dirichlet root noise (total_alpha/fraction); root policy temperature with
  early/halflife schedule, applied on *reused* roots too (the main1 reuse-root bug is a named
  regression test); tree/subtree reuse across moves; move selection by `temperature_by_ply`.
- **Continuous scheduler** (`run_continuous` semantics, `mcts.rs:694-1086`): per-game slots,
  `virtual_batch_size`, `flush_target`, `active_root_limit`, on_move Python callback carrying root
  policy, root value, and move-class flags (consumed as in `selfplay.py:1358-1370`).
- **PCR**: `pcr_full_proportion ∈ (0,1]` drawn per ply on the PCR stream → MoveClass Full/Fast;
  fast = `pcr_fast_visits`, no noise, no forced playouts, root temp 1.0 (`mcts.rs:139-207`).
- **Policy-init openings**: per game, with prob `policy_init_fraction`, the first
  `~TruncExp(policy_init_avg_plies, cap policy_init_max_plies)` plies sample the RAW root prior at
  `policy_init_temperature`, 1 visit, class Init (`mcts.rs:110-130`).
- **Deterministic seed streams**: the exact `mix_seed(base_seed, game_key, ply, stream)` hash
  (`mcts.rs:2071-2079` — specified in this doc as a written contract, splitmix-style constants and
  all) and stream ids 0–5: ROOT_NOISE, MOVE_SELECT, PCR, POLICY_INIT_SELECT, POLICY_INIT_COUNT,
  POLICY_INIT_SAMPLE (`mcts.rs:61-66`). Keeping the identical hash makes cross-implementation
  parity *testable*, not just claimed.
- **Forced playouts** `forced_playout_k` for Full roots (`mcts.rs:165-170`).
- **TSS, toggleable** via the `tss_enabled` config key (landed 2026-06-12): consulted at the three
  proven sites — tactical-candidate injection at expansion, phase-aware hitting-set leaf override,
  root move-selection guard (`threats_shared.rs:12-17`). Simplification vs dense: every tactical
  cell is always in-vocabulary (threat empties sit within distance 5 of stones,
  `threats_shared.rs:33-38`), so the call-site crop filter is deleted.
- **Lockstep `search`** driver kept alongside continuous (eval ladder / match screen / debug use).
- **Eval cache**: `HashMap<StateHash, Arc<Evaluation>>`, key = `hexo_utils::hash_state`
  (`hexo_utils/rust/src/state_hash.rs:31` — pure engine hash, no encoder dependence),
  bounded-insert ~1M entries, Arc-shared priors (`mcts_eval.rs:397-513` semantics).

**Differential parity test (the rewrite's safety net):** run hexfield search and dense_cnn search
on the same ≥100-position corpus with the same seeds and a deterministic stub evaluator (priors a
pure hash of `(state_hash, action_id)`, value a pure hash of state_hash — model-free). Identical
PUCT constants + identical seed streams + identical priors ⇒ assert *identical visit counts and
chosen moves*. This pins every scheduler/tree semantic at once, cheaply, without a GPU.

### 6.3 Evaluator payload ABI (the only new wire contract)

Request (Rust → Python), one dict per flush; CSR flat-concat over rows (variable-length precedent:
hexgt's candidate CSR, `hexgt/rust/src/mcts_eval.rs:240-300`):

| key | type / dtype | length | meaning |
|-----|--------------|--------|---------|
| `abi` | int | 1 | payload schema version = 1 (fail-loud drift guard) |
| `shape` | tuple (B, total_nodes, 13) | — | row count, node total, feature dim |
| `node_feats` | zero-copy f16 buffer | total_nodes × 13 | features, node-major, canonical (q,r) order per row |
| `node_qr` | PyBytes i16 | total_nodes × 2 | axial coords (Python builds the bias index from these) |
| `node_row_offsets` | tuple i64 | B + 1 | node CSR offsets |
| `nbr_pos` | PyBytes i32 | total_nodes × 6 | within-row position of `v + D[d]`, −1 if absent; direction order = D (§3.2) |
| `legal_pos` | PyBytes i32 | total_legal | within-row node positions of legal cells, ascending |
| `legal_row_offsets` | tuple i64 | B + 1 | legal CSR offsets |

Rust keeps the per-row `legal_action_ids` aligned with `legal_pos` order and never round-trips ids
(dense pattern, `mcts_eval.rs:354-383`; hexgt pattern, `hexgt mcts_eval.rs:250-252`). The f16
conversion happens in Rust's parallel assembly; `node_feats` crosses as a buffer-protocol view
(zero-copy PlaneBuffer pattern, `mcts_eval.rs:254-278`).

Reply (Python → Rust) — **byte-identical ABI to dense_cnn** (`mcts_eval.rs:339-357`,
`inference.py:337-412`):

```
{ "values_bytes": f32 × B,            # binned-value expectation, clamped [-1, 1]
  "priors_bytes": f32 × total_legal } # per-legal softmax, positional, row-major
```

Python internals of `evaluate_payload`: scatter rows into per-bucket padded `(B_k, Npad, F)`
layouts (§10.3), forward each bucket, gather legal logits and run the proven segment softmax with
the device-side zero-mass check (`inference.py:385-408`), reassemble in row order. Rust then zips
positionally with retained action ids, validates (finite, non-negative, unique, positive mass),
**sorts descending, normalizes to sum 1.0** → `Evaluation{value, priors: Vec<(PackedCoord, f32)>}`
(`mcts_eval.rs:515-580` semantics). Because the vocabulary is the full legal set,
`legal_action_count == priors.len()` always — the dense crop-shortfall branch is deleted.

The PUCT tree consumes `(action_id, prior)` pairs opaquely (`mcts_eval.rs` header note; brief §4),
so everything downstream of the evaluator is untouched by hexfield's representation.

---

## 7. Data pipeline

### 7.1 Shard schema — reuse compact v1, byte-compatible

Self-play writes one columnar compact `.npz` + JSON sidecar per game with **exactly the existing
compact-v1 keys** (`dense_cnn_restnet/compact_io.py:56-150`): `schema_version=1`, `num_rows`,
`horizons`, per-row scalars (`turn_index` i32, `current_player` u8, `phase` object,
`center_q/center_r` i16, `value` f32, `moves_left` f32 with −1 sentinel, `first_q/first_r` i16 +
`first_present` u8, `stvalue`/`stvalue_mask` (n,h) f32) and CSR pairs (`stones_qr/owner/off`,
`legal_ids/off`, `hist_qr/owner/idx/off`, `own_hot_qr/off`, `opp_hot_qr/off`, `last_hot_qr/off`,
`pol_act/pol_w/off`, `opp_act/opp_w/off`). Rows are RAW FACTS — representation-agnostic by design
(brief §4), which is the whole point: **hexfield's expander can read existing restnet self-play
shards unmodified** (it ignores `center_*`), giving free distillation data (§8). hexfield's writer
fills `center_q = center_r = 0` (documented as unused) and tags the sidecar `"lineage":
"hexfield"`. Read-compat is one-way (new reads old); old-reads-new is a non-goal.

The writer/reader is a fresh ~200-line implementation against this schema (compact_io is lineage
code, not shared infra); a golden cross-read test pins compatibility (§9).

### 7.2 Expand-time featurization (training read path)

Per row, in Python (DataLoader workers): draw symmetry → transform facts (§5) → build support from
stored `legal_ids` + stones (§2.1; engine truth was captured at write time, no engine needed at
train time) → BFS distances → emit:

```
input    (N, 13) f32      qr (N, 2) i32        nbr_pos (N, 6) i32 (−1 absent)
legal    (N,) bool        policy (N,) f32      opp_policy (N,) f32
value () f32              stvalue_h () f32 + mask ×3      moves_left () f32 + mask
```

Strict validation at this boundary (finite, non-negative, positive policy mass — `input.py` /
`losses.py` discipline). Python is the *primary, debuggable* featurizer for training; Rust is the
serve-time featurizer; a fixture-based parity test is the contract between them (§9). (A
Rust-only featurizer was considered and rejected: it would force a WSL maturin rebuild into every
feature-debug loop.)

### 7.3 Replay window and trainer loop

Port restnet's KataGo-style machinery semantics 1:1 (`replay.py:1-24`): policy-surprise frequency
weighting materialized as row duplication before the per-game write (loss stays unweighted);
mtime-ordered tapered shuffle window (keep 300k rows, taper 0.65); keep-prob subsample → permute →
batch-aligned output shards consumed by `select_training_samples`/`train_passes`
(`trainer.py:124,256`). PCR row filtering at the source: only Full-search rows are written, and
truncated games are quarantined under `drop_truncated_rows` (`selfplay.py:309-336`).

**Trainer deltas vs restnet (the only two):**
1. *Bucketed token-budget collate*: rows stream from the shuffled shards into per-bucket bins
   (`Npad = ceil((N+8)/256)·256`); a bin flushes as a micro-batch when adding a row would exceed the
   token budget `T = 32,768` node slots (nominal B=32 at N≈1024; B shrinks automatically for huge
   positions). One knob, bounds worst-case VRAM by a closed-form law (§10.4), zero semantic effect
   on sampling beyond bin-local ordering (rows were globally shuffled upstream).
2. *Pair-index reuse*: the (B, S, S) bias index is computed once per micro-batch, shared across the
   3 A blocks.

Everything else is identical: AdamW lr 1e-3, wd 1e-4 on matrix weights only (ndim ≥ 2, excluding
bias tables and token inits — the param-split rule of `plugin.py:72-87` extended to
`relative_bias_table` and `token_init`), AMP autocast + GradScaler, grad-clip 1.0, D6 per row,
optional weight-EMA with restnet semantics (production uses EMA 115). Checkpoints: strict-load
state_dict + optimizer + window bookkeeping + config echo; **no silent partial loads** (the main1
random-value-head lesson).

---

## 8. Bootstrap

**BC prefit from the HF corpus** (`timmyburn/hexo-bootstrap-corpus`, 6,902 decisive games ≈ 431k
positions), following the proven recipe shape of `scripts/bootstrap_dense_cnn_restnet_hf.py`:
replay each game through `hexo_engine` (legality + terminal validated, engine winner authoritative,
lines 72-128), capture a compact row per decision with policy = one-hot played action, finalize
with `finalize_game_samples` semantics: hard-z value, opp_policy = next opponent one-hot, STV
horizons = () (no search values exist → masked), moves_left = real remaining decisions (decisive
games → unmasked). Write with hexfield's writer; build the standard shuffle; train the standard
loss with policy/value/opp/moves_left active. Game-level train/val split (no game straddles).
3–5 passes; gates in §11 (M3).

**Optional distillation accelerant**: because hexfield reads compact v1 (§7.1), recent restnet
self-play shards (real MCTS visit policies + hard outcomes + root-value STV targets) can be mixed
into the prefit window with zero new code. Listed as optional, not on the critical path — the BC
prefit alone is the known-good recipe.

---

## 9. Code architecture

### 9.1 Greenfield package layout

```
packages/hexfield/
  pyproject.toml                      # deps: hexo_engine, hexo_utils, hexo_train, torch, numpy
                                      # entry point [hexo_train.models] hexfield = hexfield.plugin
  python/hexfield/
    constants.py     # F indices, VALUE_BINS=65, MOVES_LEFT_CAP=512, D order, bias-table layout
    geometry.py      # hex_distance, D6 transforms, pack/unpack action ids, EXACT_LUT builder
    support.py       # support construction + BFS distances + featurization (train-time truth)
    model.py         # HexNodeConv, ConvBlock, AttnBlock (sdpa+materialized), HexfieldNet
    losses.py        # 65-bin helpers, masked soft CE, total loss        (ports of verified math)
    samples.py       # compact row dataclass, finalize targets (z/opp/STV/moves_left)
    shards.py        # compact-v1 writer/reader (schema §7.1)
    replay.py        # surprise duplication, mtime window/taper, shuffle build
    batching.py      # bucket rule, token-budget collate, nbr/global index assembly
    inference.py     # HexfieldEvaluator: evaluate_payload ABI + direct state inference
    selfplay.py      # continuous-scheduler epoch driver (on_move, finalize, writes)
    trainer.py       # select_training_samples / train_passes / close
    evaluation.py    # anchored-ladder epoch eval (restnet semantics)
    checkpoints.py   # strict loader/saver
    player.py        # standard bot surface (runner/match-screen/debug worker)
    plugin.py        # hexo_train plugin: build_model / overrides / generate_selfplay /
                     # evaluate_epoch / calibrate_performance      (registry.py:38-58 contract)
  rust/src/
    lib.rs           # register_pybridge for hexo_models._rust.hexfield
    support.rs       # support set + features + sample-facts gen (serve-time truth)
    payload.rs       # payload assembly + reply parse/validate/sort/normalize (ABI §6.3)
    tree.rs          # PUCT node/edges, widening, FPU, virtual loss, reuse
    search.rs        # lockstep + continuous schedulers, PCR, policy-init, noise, seeds, TSS
    cache.rs         # bounded state_hash-keyed eval cache
```

Links against shared infra only: `hexo_engine` (state/legality/windows), `hexo_utils`
(`hash_state`, records), `hexo_train` (pipeline/config/diagnostics/artifacts),
`crate::threats_shared`. Zero imports from dense_cnn/restnet/hexgt Python or Rust. Config:
`configs/hexfield_main_1.toml` with `[model] module = "hexfield.plugin"` (mode-1 loading,
`registry.py:14-18`).

Build story: one `#[path]` include + registration in `hexo_models/rust/src/lib.rs` (§6.1); rebuild
via the existing canonical script (WSL venv, maturin --release). No new crate, no new build
pipeline. Tests are authoritative in the WSL venv (repo convention).

### 9.2 Parity / oracle test strategy (the contracts that keep two of anything honest)

1. **Attention oracle**: sdpa vs materialized ≤ 1e-4 abs (fp32), random shapes incl. pad rows.
2. **Conv oracle**: gather-GEMM vs per-node loop reference, incl. missing-neighbor zeros.
3. **Rust↔Python featurizer parity**: Rust `support.rs` emits golden fixtures (EncodedSupport →
   npz) over ~50 randomly played engine states; Python `support.py` must match exactly (ints) /
   ≤1e-6 (floats), including node order, nbr_pos, legal_pos, BFS distances.
4. **Search differential**: stub-evaluator visit-count equality vs dense_cnn (§6.2) for lockstep
   AND continuous (move-class sequences, chosen moves, per-move visit counts); TSS on/off toggle
   coverage; reused-root temperature regression.
5. **ABI golden**: payload construct → evaluate → parse round-trip; byte-count and validation
   failure modes (dup id, NaN prior, zero mass, terminal row).
6. **Shard compatibility**: hexfield reader on a committed restnet shard fixture; writer→reader
   round-trip; D6 commutation property tests (§5).
7. **e2e smoke**: 4-game epoch at 64 visits through the real plugin on CPU.

---

## 10. Perf budget

### 10.1 Parameters (C=96, heads=4, mlp_ratio=2) — total **1,212,675 ≈ 1.21M**

| component | formula | params |
|---|---|---|
| stem | 7·13·96 + 96 + LN 192 | 9,024 |
| conv blocks ×6 | (2·(7·96·96+96) + 2·192) ×6 | 777,600 |
| attn blocks ×3 | (4·(96²+96) + 229·4 + 2·192 + (96·192+192) + (192·96+96)) ×3 | 227,100 |
| tokens + final LN | 8·96 + 192 | 960 |
| policy + opp heads | 2·(7·96·96 + 96 + 97) | 129,410 |
| value head | 192·96+96 + 96·65+65 | 24,833 |
| aux reduction + 4 tops | 192·96+96 + 4·(96·65+65) | 43,748 |

(vs current production model ~1.5M; brief target ~1.2M.)

### 10.2 FLOPs per eval vs dense_cnn_restnet

Per-position MACs: convs ≈ 912k·N (stem 8.7k + trunk 774k + head convs 129k per node); attention ≈
221k·S + 576·S² (3 blocks: QKVO+MLP 73.7k·S, scores+AV 192·S² each). Dense restnet reference
(R_R_R_T_R_R_T_R, 41²=1681 cells, 9-tap compute): ≈ 3.30 G-MAC/eval.

| N (nodes) | hexfield G-MAC | ratio vs dense |
|---|---|---|
| 600 | 0.89 | 0.27× |
| 900 (typical mid-game) | 1.50 | 0.45× |
| 1500 | 3.01 | 0.91× |
| 3000 (tail) | 8.61 | 2.6× |

Typical games live at 0.3–0.9× dense compute; the N² attention tail dominates only beyond ~1.5k
nodes (brief §4 agrees). Non-GEMM overheads (gathers, bias index build ≈ 6 int ops ×S² per row,
once per batch) are budgeted at 10–20% and measured at M8.

### 10.3 Batching & padding plan

- **One layout**: padded `(B, Npad, ·)` per bucket; bucket rule `S_pad = ceil((N+8)/256)·256`
  (~12 live shapes; unbounded above — no cap, ever). Convs and attention share it; pad rows are
  inert by masking (§3.2/§3.4).
- **Inference**: each Rust flush (avg ≈54 leaves) is grouped by bucket inside `evaluate_payload`;
  buckets exceeding the eval token budget (24,576 slots) are chunked. 1–4 forwards per flush.
- **Training**: token budget T = 32,768 slots per micro-batch (§7.3).
- **fp16**: f16 feature transport (wire); fp16 eval weights behind the restnet-style adopt-and-gate
  pattern (fp32 fallback on gate failure); fp16-safe attention by construction (§3.4). LN/softmax/CE
  accumulate in fp32.
- **torch.compile**: optional M8 experiment (`dynamic=False` per bucket, correctness-gated);
  the model has no data-dependent Python guards on the eval path by design.
- **TRT story**: explicitly **not pursued**. Variable-length sequences + per-sample additive bias +
  gather convs make export fragile, and the measured TRT win was on the fixed-shape dense model.
  fp16 + bucketing + (optional) compile carry the perf budget.

### 10.4 VRAM

Closed-form law for the dominant attention term per micro-batch: bias masks (3 layers, fp16,
4 heads) + shared i64 pair index ≈ `(3·8 + 8)·S·T` bytes = `32·S·T`.

- **Training @ T=32,768**: S=1024 → 1.07 GB; worst bucket S=3072 → 3.2 GB. Plus saved conv gathers
  12×(T·7·96·2B) ≈ 0.53 GB, other activations ≈ 0.3 GB, params+grads+Adam ≈ 0.02 GB.
  **Peak ≈ 2–4 GB** — comfortable on 12 GB beside the evaluator. (Optional conv-block activation
  checkpointing exists as a documented contingency, default off.)
- **Inference @ eval budget 24,576, no backward**: index ≈ 0.2 GB + one transient mask ≈ 0.2 GB +
  activations ≈ 0.1 GB + fp16 weights 2.4 MB. **Peak ≈ 0.5 GB per chunk** at 256-leaf flushes.

### 10.5 Throughput expectation

At typical N≈900 the model is ~0.45× dense MACs but pays gather/bias overheads; the honest
projection is **0.6–1.2× dense evaluator wall-clock for typical positions**, degrading on marathon
games (the same games the crop used to silently truncate instead). The real cap remains
evals/position (visits × game length, owner-locked), unchanged here. M8 measures before anyone
believes a number.

---

## 11. Milestones (ordered; each gate blocks the next)

| # | deliverable | acceptance gate |
|---|---|---|
| M0 | geometry.py, support.py + property tests | support = BFS-9 equivalence on random states; halo = dist-9 shell; ply-0 = 7 nodes; D6 commutation (12/12) |
| M1 | model.py forward | param count = 1,212,675; sdpa≡materialized ≤1e-4; conv oracle exact; grads reach every named param; pad-row inertness test (outputs invariant to pad width) |
| M2 | losses, samples, shards, batching | target math matches restnet reference values on fixtures; restnet-shard cross-read; writer round-trip; collate respects budget/buckets |
| M3 | BC prefit on HF corpus | runs to completion under AMP, no NaN; val policy CE/top-1 within 5% of the dense_cnn_restnet BC reference on the same game-split; value calibration `|mean(v̂)−mean(z)|` ≤ 0.05 on val; VRAM peak ≤ 6 GB. (LN-vs-BN tripwire: gross undertraining here triggers the §3.3 contingency.) |
| M4 | Rust support.rs + sample facts | Rust↔Python parity fixtures exact; sample-facts equal dense_cnn's for the shared fields on identical states |
| M5 | payload.rs + lockstep search | ABI golden tests; stub-evaluator visit parity vs dense_cnn lockstep on ≥100 positions (exact) |
| M6 | continuous scheduler + PCR/policy-init/noise/TSS | stub parity for continuous: identical move-class sequences, chosen moves, visit counts; seed-stream vectors match `mix_seed` reference; TSS toggle test; reuse-root temperature regression |
| M7 | plugin + e2e | 4-game, 64-visit epoch through `hexo_train` on the WSL venv; shards/sidecars/diagnostics/checkpoint round-trip all present; checkpoint strict-load |
| M8 | perf calibration | measured evals/s vs dense at matched search settings; support-size and bucket histograms; fp16 gate passes; VRAM telemetry within §10.4 |
| M9 | self-play soak | 2–3 epochs end-to-end; policy entropy / game length / value calibration in sane bands; prefit-seeded bot ≥ smoke-parity vs its own BC checkpoint in a 100-game arena; handoff doc |

(GPU scheduling vs the live main_4 run is out of scope per the brief.)

---

## 12. Envelope deviations

§2 owner-locked items: **no deviations; no contradictions found.**

§3 refinements (with justification):

1. **LayerNorm everywhere; BatchNorm knob deleted** (default said BN with LN fallback). Variable-N
   batches make BN stats composition-dependent and pad-sensitive; LN removes running stats, conv-BN
   folding, and train/eval divergence, and unifies with the attention blocks' norm. Contingency
   (masked BN1d) is documented behind the M3 gate instead of shipped as a knob. (§3.3)
2. **Hot features stay at the trusted count ≥ 4** (brief text said "≥count-3"): matches dense_cnn's
   verified `fill_hot_cells` (`encoding.rs:226-266`) and makes the feature threshold identical to
   the TSS threat definition — one "threat window" concept repo-wide. (§2.2)
3. **Stem pinned to a single 7-tap HexNodeConv + LN + ReLU** (brief left the stem open): reuses the
   one conv primitive, matches EmbedNet semantics. (§3.1)
4. **Token-budget micro-batching** added to the trainer (brief said batch 32): batch 32 remains the
   nominal row count at typical N, but on an unbounded board a row-count batch has unbounded memory;
   the budget is the single knob that restores a closed-form VRAM bound. (§7.3, §10.4)
5. **Per-sample bias materialization accepted** (and budgeted) rather than adopting flex-attention
   or windowing: it is the simplest exact implementation of the locked rel-pos scheme; the shared
   pair-index + token budget keep it bounded. Flagged as the primary perf risk. (§3.4, §10)
6. Adopted unchanged: 9-layer `C C C A C C A C A`, token split 2/2/4, 13-feature port, 229-row bias
   table (217+8+1+3), channels 96 / 4 heads / mlp_ratio 2, sdpa+materialized oracle, 65-bin heads,
   production loss weights, KataGo window 300k/0.65, AdamW 1e-3/1e-4, grad-clip 1.0, batch-32
   nominal, hard-z, D6-by-augmentation.
