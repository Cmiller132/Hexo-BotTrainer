# hexline — a hex-native, support-set lineage (design competition entry: GEOMETRY-FIRST)

Designer lens: **hex-native representation purity**. Every representational choice below is derived
from the game itself — the axial lattice, the three win axes, the length-6 window, the D6 group,
the radius-8 legality rule — and nothing references a square grid, a crop window, or an absolute
coordinate frame. Where the brief leaves a default refinable, the refinement is accepted only if it
is *forced by the geometry*; otherwise the trusted dense_cnn semantics are kept verbatim.

---

## 1. Identity

**hexline** (the length-6 line is the game's atom; the lineage is built outward from it).

Thesis: the model's domain is exactly the game's *support set* — the union of radius-8 disks around
the stones (= stones ∪ legal) plus its 1-ring halo — represented as an unordered set of hex cells
carrying axial coordinates. Local reasoning is a direction-typed 7-tap hex convolution
(mathematically the same family as dense_cnn's masked 3×3, with the square-grid carrier removed);
global reasoning is pre-norm multi-head attention over the node set with a learned relative-position
bias keyed on *true axial offsets*, exact out to hex radius 8 and axis-aware beyond it; eight
carried-through summary tokens give the value/aux heads a position-free global readout. Features
that the dense lineage flattened into binary planes are restored to their native, per-axis form
(line-potential channels per win axis per player), because the win condition is literally per-axis
and D6 acts on those channels by an exact permutation. There is no crop anywhere: every legal cell
has a policy logit by construction, the frozen-win failure class of main_3 is unrepresentable, and
the model's cost scales with the true size of the position instead of a fixed 41×41 carrier.
Trunk depth 9 (`C C C A C C A C A`), 96 channels, ≈1.214M parameters.

---

## 2. Input representation

### 2.1 Support set construction

Definitions (engine truth; `hexo_engine/rust/src/coord.rs:77-82` hex distance,
`legal.rs` legal store, LEGAL_RADIUS=8):

- `stones` = occupied cells (both colours).
- `legal` = engine legal set = { empty cells within hex-distance 8 of any stone }
  (ply 0 special case: `legal = {(0,0)}`, the forced origin).
- `core = stones ∪ legal`. Because every stone is placed within distance 8 of an existing stone,
  `core` is exactly `∪_s disk8(s)` and is always **connected**.
- `halo` = { c ∉ core : c is a hex neighbor of some cell in core }, computed by one 6-neighbor
  expansion of `core` minus `core`. Property (provable from the disk-union form, and unit-tested):
  in every position after ply 0, `halo` = the set of empty cells at hex distance **exactly 9** from
  the nearest stone. At ply 0 (no stones), `halo` = the 6 neighbors of the origin.
- **Support set** `S = core ∪ halo`. `|S|`: ply 1 → 217 core + 54 halo = 271; mid-game ≈ 650–1600;
  long spread games up to ≈3.3k (brief §4 scale plus the ring).

Node identity & ordering: nodes are the cells of `S`, **sorted ascending by packed key**
`pack(q,r) = ((q+2^15)<<16)|(r+2^15)` (the universal action id, `d6.py:41-45`). Ascending-packed
order = lexicographic (q, r); it is identical to compute in Rust and numpy (sort of a u32 vector),
which makes Rust↔Python encoder parity a byte-compare. `coord → node index` is a hash map (Rust) /
`np.searchsorted` on the sorted key vector (Python).

Role flags per node: `stone` (own/opp), `legal`, `halo`. Halo cells carry features and participate
in conv + attention but **never receive logits** (owner-locked §2.1). Cells outside `S` do not
exist: a 7-tap conv tap that falls outside `S` contributes zeros (zero-padding semantics, §3 brief
default), which is exact because every out-of-support cell is empty and far (≥10) from all stones —
its true feature vector is ~0 anyway except the `empty` marker, and the halo ring already gives the
boundary convs one cell of real "empty but unplayable" context.

Canonical direction order (used for conv taps and everywhere a direction is enumerated):

```
DIR[0]=(+1, 0)  DIR[1]=( 0,+1)  DIR[2]=(-1,+1)  DIR[3]=(-1, 0)  DIR[4]=( 0,-1)  DIR[5]=(+1,-1)
```

This is the rotate60 orbit of (+1,0) (`d6.py:129-131`: rot60(q,r) = (−r, q+r)), so a 60° rotation
maps DIR[k] → DIR[k+1 mod 6] — the cleanest possible statement of the lattice's symmetry. The three
win axes in this basis: Q = ±DIR[0], R = ±DIR[1], QR = ±DIR[5] (`tactics.rs:23-53`).

Neighbor table: `nbr[i, k] = index of cell (p_i + DIR[k]) in S, else SENTINEL` (sentinel = `|S|`,
the index of an appended all-zero "dump row"; this is the same trick restnet's content scope uses,
`architecture.py:637-641`).

### 2.2 Node feature table (F = 17 channels, fp32 at train expansion, fp16 on the wire)

| # | name | range | exact formula / source |
|---|------|-------|------------------------|
| 0 | own_stone | {0,1} | cell occupied by current player |
| 1 | opp_stone | {0,1} | cell occupied by opponent |
| 2 | empty | {0,1} | `1 − own − opp` (all legal + halo cells). Native role: support-membership marker — distinguishes "empty cell exists here" from the zero contribution of out-of-support taps |
| 3 | legal | {0,1} | cell ∈ engine legal set (logit-bearing nodes) |
| 4 | phase_second | {0,1} const | 1 iff phase == SecondStone (broadcast to all nodes) |
| 5 | first_stone | {0,1} | 1 at the cell of the current turn's first placement (SecondStone phase only, else all-0) |
| 6 | player_colour | {0,1} const | 1 iff current player == player0 |
| 7 | own_recency | (0,1] | at own stones: `1/(1 + latest_idx − placement_idx)`; max kept on duplicates (mirrors `input.py:122-137`) |
| 8 | opp_recency | (0,1] | same for opponent stones |
| 9–11 | own_line_q, own_line_r, own_line_qr | {0, .2, .4, .6, .8, 1} | per win axis a: `max(count/5)` over all *own-active* length-6 windows through this cell along axis a, scattered to the window's **empty** cells (see 2.3) |
| 12–14 | opp_line_q, opp_line_r, opp_line_qr | same | same for opponent-active windows |
| 15 | dist_nearest | [0, 1.125] | `min_s hexdist(c, s) / 8`; stones → 0; legal → d/8 ∈ [1/8, 1]; halo → 9/8 = 1.125 (a natural "beyond the rim" marker). Empty board (ply 0 only): 0 everywhere |
| 16 | opp_last_turn | {0,1} | 1 at the ≤2 cells of the opponent's last completed turn (`encoding.rs:268-298` semantics) |

Notes:

- Features 0–8, 15–16 are the native port of the trusted 13 planes; the crop-center-distance plane
  is replaced by dist-to-nearest-stone exactly as the brief's §3 default specifies.
- **Features 9–14 replace the two binary `own_hot`/`opp_hot` planes** (deviation, §12.1). dense_cnn's
  hot = empty cells of single-colour count≥4 windows (`encoding.rs:226-266`), i.e. the engine threat
  predicate (`tactics.rs:188-198`) collapsed across axes and thresholded. The native form keeps the
  *axis identity* (the win condition is per-axis; two count-3 windows crossing at a cell on
  different axes is the textbook double-threat pattern, invisible to an axis-collapsed plane) and
  the *count grade* (count 5 = win-in-one at that cell; count 4 = engine threat; count 3 = one full
  turn from threat — the Connect6 tempo unit is 2 placements, so 3/4/5 are tactically distinct
  classes). The information strictly contains the trusted planes (hot ≡ any own_line_* ≥ 0.8). The
  `placements_made < 7` gate in `encoding.rs:232` is dropped: it suppressed early *binary* noise;
  graded counts carry their own salience (early windows read 0.2–0.4) and a move-count gate is not a
  geometric quantity. Window empty cells are always within distance 5 of a stone in the window, so
  line features land only on legal nodes — never on halo.
- fp16 wire safety: every feature is a small dyadic-or-near value; the recency reciprocals are
  within 5e-4 of their fp16 rounding — same precision class dense_cnn already ships as f16 planes
  (`mcts_eval.rs:270-276` "f16 is loss-free for search (gated)").

### 2.3 Line-potential construction (engine-exact)

From the engine's incremental window store (`tactics.rs:343+`, entries = every length-6 window
containing ≥1 stone): for each entry `w` with `active_player(w) = p` (single-colour,
`tactics.rs:172-186`) and `count = w.count(p) ∈ 1..5`:

```
for each empty cell c of w:                     # w.empty_cells(), ≤ 5 cells
    ch = (p == current ? own : opp) line channel of w.axis
    feat[c, ch] = max(feat[c, ch], count / 5)
```

Cost: one pass over the window store (≤ ~18 × stones entries), already how `fill_hot_cells` works.

### 2.4 Edge cases

- **Ply 0** (empty board, P0 to move): `S` = {origin} ∪ 6 halo cells = 7 nodes, 1 legal node;
  all stone-derived features 0; dist_nearest defined 0. The forward is well-defined (search is
  trivially forced here, but the contract holds).
- **Ply 1** (1 stone): 217 core + 54 halo = 271 nodes, 216 legal.
- Support connectivity is guaranteed (2.1), so the BFS in 2.5 needs no multi-component handling.
- No fact can ever be "out of crop": stones, history, policy mass, legal cells are all in `S` by
  construction. The spill-counting machinery (`samples.py:285-307`) is obsolete for self-features;
  only the *opp-policy* target can spill (see §4.2) and that is counted in the sidecar.

### 2.5 Derived tensors built at encode time

- `nbr (|S|, 6) i32` — via 6 shifted lookups (Rust hash map / numpy searchsorted on packed keys).
- `dist_nearest` — multi-source BFS from the stone set over `S`, ≤ 9 rounds of vectorized
  neighbor-set intersection (support is connected; halo terminates the frontier).
- `legal_index (K,) i32` — node indices of legal cells **in the engine's legal-id emission order**
  (the positional prior contract, §6).

---

## 3. Trunk

Channels `C = 96`, heads `H = 4` (head_dim 24), `mlp_ratio = 2`, dropout 0. All node states live in
a **flat layout** `(M, C)` where `M = Σ_g |S_g|` over the batch (CSR with `row_offsets (B+1) i64`);
attention blocks construct padded per-graph views on the fly (3.4). Tokens live separately as
`(B, 8, C)`.

### 3.1 Stem

`x = ReLU(BN(HexConv7(feat)))` — one direction-typed 7-tap conv 17→96, bias-free, then BatchNorm.
Receptive radius after the stem: 1.

### 3.2 The direction-typed 7-tap hex conv (the local primitive, owner-locked §2.3)

For node i:

```
y_i = b + W_self · x_i + Σ_{k=0..5} W_k · x̂[nbr[i,k]]          x̂ = cat([x, zeros(1,C)])
```

One full `C_out×C_in` matrix per relative direction plus center, shared at every node — exactly
dense_cnn's `HexConv2d` family with the carrier removed: the masked 3×3's seven surviving taps
(`architecture.py:163-179`) are in bijection with {center} ∪ DIR (kernel offsets (drow,dcol) =
(dr,dq) with the (−1,−1)/(1,1) corners masked). Implementation: gather the 7 neighborhoods,
concatenate to `(M, 7C)`, one GEMM with weight `(C_out, 7C)` — a single cuBLAS call, no cuDNN, no
spatial padding waste. **Equivalence test** (M1 gate): on a synthetic full-disk support, this op
with weights copied tap-for-tap equals dense_cnn's `HexConv2d` on the corresponding zero-padded
41×41 tensor, bit-for-bit in fp64.

### 3.3 Conv residual block (`C`) — 6 of them

dense_cnn post-activation family (brief §3 default):

```
h = ReLU(BN1(Conv7_1(x)))
h = BN2(Conv7_2(h))
x = ReLU(x + h)
```

Norm choice: **BatchNorm1d over the flat node dimension** (statistics per channel over all real
nodes in the batch; the dump row is appended inside each conv call and never enters BN). Why BN and
not LN here: (a) it is the trusted dense_cnn block family; (b) it folds into the conv GEMM at
inference (affine fold, mirroring `fuse_conv_bn_eval` usage at `architecture.py:914-920`) so eval
cost is exactly 7 gathers + 1 GEMM per conv; (c) flat-node BN sees *only real cells* — better
statistics than dense BN, which averaged over ~52% permanently-empty crop corners. Batch statistics
pool ≈ 32 rows × ~900 nodes ≈ 29k samples per channel — more stable than dense's 32×1681.
`norm="bn" | "ln"` is a config knob (LN = per-node normalization, shape-independent) kept as the
documented fallback if BN under micro-batched grad accumulation (§7.4) misbehaves; BN is the
default.

Per-block params: 2·(7·96·96) + 2·(2·96) = 129,408.

### 3.4 Attention block (`A`) — 3 of them (owner-locked §2.5–6)

Pre-norm transformer block over the joint sequence `[8 tokens ; L cells]`:

```
t = LN1([tok ; cells_padded]);  s = s + MHSA(t);  s = s + MLP(LN2(s))
```

- MHSA: q/k/v/out projections `Linear(96→96)` each (bias yes), scale `1/√24`, softmax over keys,
  additive relative-position bias (below). Primary impl: `F.scaled_dot_product_attention` with the
  bias as `attn_mask`; **materialized oracle** impl (explicit `QKᵀ·scale + bias → softmax → ·V`)
  shares the same parameters and is the test reference (restnet's proven dual-path pattern,
  `architecture.py:262-281,484-497`).
- MLP: `Linear(96→192) → GELU → Linear(192→96)`.
- Padded layout: per micro-group (see §7.4/§10.3), `pad_index (G, Lp) i64` gathers flat cells (dump
  row for padding) into `(G, Lp, C)`; tokens are concatenated in front → sequence length `8+Lp`.
  After the block, cells scatter back to flat via the same index (padding discarded); **tokens
  persist** in their own `(B, 8, C)` tensor across the whole trunk and pass through `C` blocks as
  identity (they have no position; a geometric op may not touch them).
- Padded-key masking: additive `−3.0e4` on padded key columns. Deliberately **not** −1e9: the model
  runs pure-fp16 at inference and −1e9 saturates to −inf in fp16 (the documented latent hazard at
  `architecture.py:89-97`); −3e4 is representable in fp16 and, after the softmax row-max subtraction,
  underflows to exactly 0 attention. Queries (including padded rows) are never masked, so no row is
  ever all-masked.

#### Relative-position bias (the geometric heart)

Per A-layer learned table `T (237, 4) fp32`, zero-init. Index function for a query node at `p_q`
and key node at `p_k`, with `(dq, dr) = p_q − p_k` (restnet's query−key convention,
`architecture.py:324-328`) and `d = max(|dq|, |dr|, |dq+dr|)`:

```
d ≤ 8        → idx = EXACT[(dq+8)·17 + (dr+8)]          # 217 exact offsets (radius-8 hex disk)
9 ≤ d ≤ 16   → on_axis = (dq==0 or dr==0 or dq+dr==0)   # collinear with a win axis
               idx = 217 + (0 if on_axis else 8) + (d−9) # 8 on-axis rings + 8 off-axis rings
d ≥ 17       → idx = 233                                 # far bucket
query=token, key=cell  → idx = 234
query=cell,  key=token → idx = 235
query=token, key=token → idx = 236
```

Table layout: rows 0–216 exact (the (17×17) axial rhombus has exactly 217 cells with d≤8; the 72
rhombus corners have d 9–16 and are unreachable in the exact branch), 217–224 on-axis rings d=9..16,
225–232 off-axis rings, 233 far, 234–236 token classes. 237 rows × 4 heads = 948 params/layer.

Why this scheme is the right geometry:

- **Exact within radius 8**: all win-window pair interactions live at d ≤ 5 (two cells share a
  length-6 window iff collinear and d ≤ 5); the exact zone covers them with margin equal to the
  legality radius — the two distances the game defines.
- **Axis-aware far field** (deviation §12.3, +8 entries/head): at d = 9–16 two stones can still
  interact through *chains of overlapping windows along one axis* (two windows sharing a cell span
  ≤ 11 cells); collinearity is the only direction information that matters out there, and
  "collinear with a win axis" is a **D6-invariant predicate** (D6 permutes the three axes —
  derivation in §5), so ring×{on,off} classes are exactly stable under augmentation. Pure rings
  (the brief default) erase the lattice's anisotropy precisely where the game is anisotropic.
- **Token rows are position-free scalars** per head (owner-locked §2.6) — one shared salience per
  relation class, content does the rest.
- D6 note: the far-field classes (217–236) are *literally invariant* under all 12 symmetries (hex
  distance and axis-collinearity are D6-invariant); only the 217 exact entries need augmentation to
  learn their symmetry ties. The geometry does most of the symmetry work for free.

Runtime: per forward and micro-group, compute `relidx (G, Lp, Lp) i32` once from padded coords
(int ops: two broadcasts, abs/max, where) and reuse it across all 3 A layers (each layer gathers its
own table: `bias = T[relidx] → (G, Lp, Lp, 4) → permute`). Memory accounting in §10.3.

#### Token mechanics (owner-locked §2.6)

8 learned init vectors `(8, 96)` (trunc_normal 0.02). At the **first** A layer the batch-broadcast
inits join the sequence; tokens participate in every A layer, bidirectionally (full joint
attention); they are untouched by C blocks. Head wiring (brief §3 split, kept): tokens 0–1 → main
value MLP; tokens 2–3 → aux (STV + moves-left) MLP; tokens 4–7 uncommitted hub capacity. Trunk ends
on an A layer, so token states are one attention round fresh at readout.

### 3.5 Interleave and per-layer rationale (brief §3 default kept: `C C C A C C A C A`)

| layer | kind | rationale |
|---|---|---|
| stem | Conv7 | feature mixing + first ring of context (radius 1) |
| 1–3 | C | local pattern algebra; receptive radius 1+6 = 7 ≥ window span 5 before any global mixing — attention layer 4 receives nodes that already *know their lines* |
| 4 | A | tokens join; first global round (cells→tokens summarize, tokens→cells broadcast) |
| 5–6 | C | digest the global context locally (threat-response patterns are local edits to global plans) |
| 7 | A | second token round (the hub needs ≥2 rounds to route cell↔cell through tokens; owner note) |
| 8 | C | final local sharpening before readout |
| 9 | A | third round; trunk ends on A so tokens and cells are maximally fresh for heads |

---

## 4. Heads & losses

All five heads (owner-locked §2.8; **no spatial ownership/win-window head**). Loss weights are the
production values (brief §4): policy 1.0, value 1.0, opp_policy 0.25, stvalue 0.1 (each), moves_left
0.1. Value targets are **hard z** with 65-bin adjacent-bin soft encoding
(`losses.py:33-53` semantics reused exactly; bins = `linspace(−1, 1, 65)`).

### 4.1 policy (per-node, legal cells only)

Head: `Conv7(96→96, bias) → ReLU → Linear(96→1)` per node → `logit (M,)`. Serve/train both consume
it through `legal_index` gathers — legality masking is *structural* (gather = mask), zero coverage
loss by construction (owner-locked §2.4).

Training loss: segment cross-entropy over each row's legal set:

```
ℓ_row = − Σ_k t_k · ( z_k − logsumexp_{j ∈ legal(row)} z_j )      t = visit weights / Σ
```

implemented with the scatter-amax/scatter-add pattern proven at `inference.py:385-403`, computed in
fp32. Target = MCTS visit policy (action-id→weight), mapped action→node by packed key; every target
action is a legal node by construction (fail-loud check kept, `losses.py:84-94` semantics).

### 4.2 opp_policy (auxiliary, weight 0.25)

Second head, identical structure. Target = the opponent's *next-decision* MCTS policy. Native
support subtlety: the opponent's legal set after our (unknown at sample time) placements can extend
beyond the current support; target mass is projected onto **current legal nodes** and out-of-domain
mass is dropped (row renormalized; `allow_empty` rows contribute zero loss — mirrors
`losses.py:153-158` and the dense out-of-crop drop). Dropped mass fraction is written to the shard
sidecar as `opp_spill_mass` (telemetry, replacing `count_spill`).

### 4.3 value (main head, 65 bins, dedicated tokens — owner-locked §2.6/§2.8 separation)

`v_logits = Linear(96→65)( ReLU(Linear(192→96)( concat(tok0, tok1) )) )` — a private MLP reading
tokens 0–1 **only**; no parameter or embedding is shared with aux heads (the heads_v3 lesson applied
natively: the separation is by token + by MLP, not by a split of one pooled embedding).
Target: z ∈ {−1, +1} from the current player's perspective → `scalar_to_binned_target` →
soft-CE (`losses.py:100-131`). Truncated games: rows are **not written at all** (adopting main_4's
C2 `drop_truncated_rows` — z=0 flood is confirmed poison; configs/dense_cnn_restnet_main_4.toml
change 5), so the value target is always ±1.

### 4.4 stvalue_2 / stvalue_6 / stvalue_16 (aux, weight 0.1 each)

`aux_emb = ReLU(Linear(192→96)( concat(tok2, tok3) ))`; per horizon `Linear(96→65)`.
Targets: per-horizon EMA of future root values stepped over full turns (even decision offsets, decay
λ = (m−1)/(m+1)) — the de-aliased construction at `samples.py:357+` reused exactly, with per-row
per-horizon masks. Loss = masked binned CE (`binned_value_loss` semantics).

### 4.5 moves_left (aux, weight 0.1)

`Linear(96→65)` on `aux_emb`. Shards store the **raw** decisions-remaining scalar; the cap
(MOVES_LEFT_CAP = 512) and the affine map `[0,cap] → [−1,1]` are applied at expansion
(`constants.py:13-27` + `samples.py:277-281` semantics — re-cap without rewriting shards). Masked
where absent.

### 4.6 Forward surfaces

`forward(x) → {policy, opp_policy, value, stvalue_2, stvalue_6, stvalue_16, moves_left}` (training);
`forward_policy_value(x) → {policy, value}` (search; aux heads and their MLPs never execute at
serve time — mirrors `architecture.py:857-864`).

---

## 5. Symmetry — D6 by training-time augmentation (owner-locked §2.9)

The 12 transforms are the axial maps of `d6.py:79-136` about center **(0,0)** (the game's true
anchor: the opening is forced at the origin; the model itself is translation-invariant by
construction — convs are relative, bias is relative — so the augmentation center is a free choice
and origin keeps the math centerless). Per training row, one symmetry s ∈ {0..11} is sampled and
applied **to the raw facts before node building**, so the s=0 and s≠0 expansion code paths are
identical (dense_cnn's transform-then-project discipline, `input.py:79-108`).

What transforms, exactly:

1. **All coordinate facts**: stones, placement history, first_stone, opp_last_turn cells, legal
   action ids, policy/opp-policy action ids — `transform_coord/transform_action_id` (reused via a
   hexline-local copy of the 20-line axial map; `d6.py:129-136`: rot60 (q,r)→(−r, q+r), reflect
   (q,r)→(q, −q−r), index ≥6 = reflect-then-rotate).
2. **Window facts** (the only direction-typed shard field): each stored window contributes
   (owner, axis, count, empty-cell coords). Cells transform as coords; the **axis relabels** by the
   permutation σ_s below. No bit-order/canonical-start headache: we store and transform the empty
   cells explicitly (§7.1).
3. **Nothing else**: nbr tables, rel-pos indices, dist BFS, node order are *recomputed* from the
   transformed coordinates — they are functions of the geometry, not stored facts.

Axis action σ_s (derived from the generator maps; rotations act as the 3-cycle, reflections as a
transposition — verified against the basis vectors in §2.1):

| s (d6 index) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q → | Q | R | QR | Q | R | QR | QR | Q | R | QR | Q | R |
| R → | R | QR | Q | R | QR | Q | R | QR | Q | R | QR | Q |
| QR → | QR | Q | R | QR | Q | R | Q | R | QR | Q | R | QR |

(rotation by 60° is the cycle Q→R→QR; rotations have axis-order 3, so s∈{0,3},{1,4},{2,5} share
columns; reflection s=6 swaps Q↔QR and fixes R; s=7 fixes Q, swaps R↔QR; s=8 swaps Q↔R.)

Consequences, stated as testable properties (M2 gate):

- The 6 line-potential channels permute exactly: `feat'[σ_s(a)] = feat[a]` per player — and since
  expansion *scatters from transformed window facts*, this happens implicitly; the test asserts it
  explicitly against a permuted-reference expansion.
- The rel-pos far-field classes (rings, on/off-axis, far, token rows) are invariant: hex distance is
  D6-invariant and σ_s maps win axes to win axes, so only the 217 exact entries see permuted data.
- Conv weights and bias tables are **not** transformed (no architectural tying — owner-locked:
  augmentation, not invariance constraints). The D6-consistency probe (mean policy KL between
  `f(T_s x)` and `T_s f(x)` over a fixed validation set) is a tracked diagnostic, not a constraint.

---

## 6. Search integration

### 6.1 Evaluator payload (Rust → Python), schema `hexline/1`

One dict per evaluation chunk (chunk = the deduplicated unique states of a leaf batch, encoder
rayon-parallel like `mcts_eval.rs:247-251`). Flat-concat CSR; **no padding on the wire**; rows in
caller order (no Rust-side sorting — all grouping is Python-internal, §6.2).

| key | type | shape / len | notes |
|---|---|---|---|
| `schema` | int | — | 1 (strict-checked) |
| `shape` | tuple | (B, F=17) | row count + feature count for validation |
| `node_feat` | zero-copy f16 buffer | M_total × 17 | node-major; the `PlaneBuffer` pattern (`mcts_eval.rs:254-279`) |
| `node_q_bytes`, `node_r_bytes` | PyBytes i16 | M_total each | axial coords (rel-pos bias + debug) |
| `nbr_bytes` | PyBytes i32 | M_total × 6 | **batch-flat** neighbor indices (already offset by the row's node base), −1 = missing |
| `node_row_offsets` | tuple i64 | B+1 | CSR over nodes |
| `legal_index_bytes` | PyBytes i32 | K_total | batch-flat node index of each legal action, rows contiguous, **in the engine legal-id emission order** |
| `legal_row_offsets` | tuple i64 | B+1 | CSR over legal entries |

Rust keeps, per row, the `Vec<PackedCoord>` of action ids in the same emission order (exactly the
role of `legal_flat_indices` + kept ids in dense_cnn, `mcts_eval.rs:286-332`).

**Return contract — byte-identical to dense_cnn** (`mcts_eval.rs:339-357`):
`{"values_bytes": f32 × B, "priors_bytes": f32 × K_total positional}`. Rust zips positionally with
its kept action ids, then validates / dedups / descending-sorts / normalizes via a from-scratch
`finalize_priors` with the exact semantics of `finalize_model_priors` (`mcts_eval.rs:515-580`),
including the terminal-row and zero-mass fail-louds and the pre-normalization invariant.

### 6.2 Python evaluator (`hexline.inference.HexlineInference`)

1. Strict byte-length checks (the `require_exact_bytes` discipline, `mcts_eval.rs:582-599`).
2. `torch.frombuffer` views; upload once.
3. **L-bucket grouping** (Python-internal): rows are grouped by padded length
   `Lp ∈ {256, 384, 512, 768, 1024, 1536, 2048, 3072, then ×ceil/256}`; groups larger than the bias
   budget split into sub-groups of `G_max(Lp) = max(1, ⌊BIAS_BUDGET / (H·2·(Lp+8)²)⌋)` rows
   (BIAS_BUDGET default 192 MiB). Per group: gather flat node ranges (`index_select`), rebase `nbr`
   with a remap vector (−1 → dump row), build `pad_index`, forward `forward_policy_value`.
4. Legal priors: gather `policy_flat[legal_index_rebased]`, segment softmax over rows
   (scatter-amax/scatter-add — the proven `inference.py:385-403` pattern), f32.
5. Values: `decode_binned_value(v_logits).clamp_(−1, 1)` f32 (`inference.py:366-372` rationale).
6. Reassemble both outputs into original row order; return bytes.

Padding never leaks: padded keys are −3e4-masked, padded query outputs are dropped at scatter-back,
and convs run on the flat (unpadded) layout.

### 6.3 Cache & hashing

`HashMap<StateHash, Arc<RustEvaluation>>` keyed by `hexo_utils::hash_state`
(`hexo_utils/rust/src/state_hash.rs:31` — pure engine board hash, encoder-independent), bounded
~1M entries (dense constant `MODEL1_EVAL_CACHE_MAX_STATES` semantics). Caller-order slot fill,
duplicate-miss dedup — the `evaluate_model1_state_refs_cached` algorithm (`mcts_eval.rs:415-513`)
reimplemented as-is.

### 6.4 TSS toggle

Link `crate::threats_shared` (the shared pure-geometry core, `hexo_models/rust/src/lib.rs:21-27`)
with the `tss_enabled` config key (already landed; `dense_cnn/rust/src/mcts.rs:102,510,559`):
leaf value override + root move-selection guard, default on, toggleable per config. One structural
improvement falls out for free: dense_cnn's TSS had to skip out-of-crop tactical cells
(the frozen-win hole, `mcts_tree.rs:885-889` per main_4's post-mortem) — in hexline **every legal
cell is in-domain**, so the TSS override needs no domain check at all; the failure class is
unrepresentable. `frozen_win_override` (main_4 C3) is therefore not ported — there is nothing for
it to do.

### 6.5 Search semantics: fresh implementation, not extraction (decision + justification)

The PUCT tree, scheduler, and evaluator bridge are written from scratch in the new
`hexline` Rust module, preserving these proven semantics verbatim (checklist, with their dense_cnn
reference homes):

- batched PUCT with virtual loss; prior-sorted lazy edge materialization; nucleus widening
  (policy_mass 0.95 / max_children 96 / min_children 2); FPU (+ `root_fpu_zero_under_noise`);
  Dirichlet root noise; forced playouts; tree/subtree promotion+reuse keyed by game key and root
  hash (`mcts.rs` module doc + knob surface at `mcts.rs:379-416` — the full signature is the
  semantic contract to reproduce);
- continuous scheduler (`run_continuous`, ContinuousSlot phases Active/AwaitRootEval/Empty),
  active-root limit, PCR Full/Fast/Init move classes, policy-init openings, and the **six
  deterministic mix_seed streams** (`mcts.rs:60-80`: root_noise / move_select / pcr /
  policy_init_select / policy_init_count / policy_init_sample);
- transposition-cached evals (6.3); terminal handling; PCR fast-search non-recording.

Why fresh-write rather than extracting a shared search crate: (1) the brief mandates greenfield
code and forbids forking; (2) extraction would genericize `dense_cnn/rust/src/mcts*.rs` **while
main_4 is live on it** — touching the active lineage's search for a new lineage's convenience is
the wrong risk trade; (3) the tree is already model-agnostic at the contract level (consumes
`(action_id, prior)` pairs opaquely — brief §4), so the rewrite is a contained, well-specified
port of algorithms, not invention. Semantic parity is *demonstrated*, not assumed: an M5 harness
runs dense_cnn's search and hexline's search over the same positions with a **scripted
deterministic evaluator** (fixed priors/values injected through each side's payload ABI) and
asserts identical visit distributions, chosen moves, and widening/noise behavior at fixed seeds.

---

## 7. Data pipeline

### 7.1 Shard schema `geo_compact v1` (reuses the compact *concept*: raw facts, columnar npz + JSON sidecar, one per game)

Fixed per-row arrays: `turn_index i32`, `current_player u8`, `phase u8` (0=FirstStone,
1=SecondStone — *not* an object column; dtype hygiene deviation §12.5), `value f32`,
`moves_left f32` (raw decisions-remaining; −1 = masked), `first_present u8`, `first_q/first_r i16`,
`stvalue (N,3) f32` + `stvalue_mask (N,3) f32`.

Variable-length fields, each as (data array, `int64` offsets of len N+1) — the `compact_io.py`
layout discipline:

| field | data dtype | per element |
|---|---|---|
| stones | qr interleaved i16 ×2 + owner u8 | one stone |
| legal | u32 | action id |
| history | qr i16 ×2 + owner u8 + placement_index i32 | one placement (phase/first dropped — recency needs owner+index only) |
| windows_meta | owner u8, axis u8, count u8, n_empty u8 | one active window (engine truth at sample time) |
| windows_cells | qr i16 ×2 | empty cell of a window (second-level offsets over windows) |
| policy | action u32 + weight f32 | one visited action |
| opp_policy | action u32 + weight f32 | one action |
| opp_last_turn | qr i16 ×2 | ≤2 cells |

No `center` field exists — there is no crop. Sidecar JSON: row counts, surprise stats,
`opp_spill_mass`, lineage tag `"model": "hexline"` (dashboard contract).

### 7.2 Expand-time featurization (train read path)

Per row: sample s ∈ {0..11} → transform facts (§5) → build support set + node order (§2.1) →
features (§2.2: scatter stones/recency/last-turn; window scatter §2.3; BFS dist §2.5) → `nbr` →
`legal_index` → policy/opp targets gathered to legal positions → scalar targets. All numpy
(packed-key sort + searchsorted; no per-cell Python loops); budget ≈ a few ms/row, hidden behind
`DataLoader(num_workers=2, pin_memory=True)`.

### 7.3 Collate (batch dict, exact)

`node_feat (M,17) f32` · `node_q/node_r (M,) i32` · `nbr (M,6) i64` (already batch-flat, −1→M) ·
`row_offsets (B+1) i64` · `legal_index (K,) i64` + `legal_row_offsets (B+1) i64` ·
`policy_target (K,) f32` (aligned to legal_index) · `opp_target (K,) f32` (same alignment, dropped
mass renormalized, may be all-zero) · `value (B,) f32` · `stvalue_h (B,) f32` + masks ·
`moves_left (B,) f32` + mask.

### 7.4 Trainer loop deltas (vs the reused restnet recipe)

Reused as-is (semantics; brief §4): AdamW lr 1e-3, wd 1e-4 on matrix weights only (decay group =
conv/linear weights; **no decay** on biases, norms, rel-pos tables, token inits), AMP autocast +
GradScaler, grad-clip 1.0, batch 32 rows, per-row D6, KataGo mtime-ordered tapered shuffle window
(300k rows, taper 0.65), policy-surprise row duplication at finalize (`replay.py` docstring
contract), per-game npz + sidecar writes.

New: **L-bucket micro-batching with weighted gradient accumulation** — the 32-row batch is split
into bucket groups (same ladder/budget as §6.2); per group g: forward, `loss_g · (n_g/32)`,
`scaler.scale(...).backward()`; one optimizer step per batch. Mathematically the same objective as
a monolithic batch (losses are row-means reweighted to the global mean); BN sees per-group
statistics (pool ≈ n_g·L̄ ≈ thousands of nodes — ample; LN knob is the escape hatch). Shuffle
output stays fully random — rows are *not* grouped by length at write time (that would correlate
game phase within batches); grouping is a runtime detail.

---

## 8. Bootstrap

### 8.1 BC prefit from the HF corpus (primary)

`scripts/bootstrap_hexline_hf.py`, the `bootstrap_dense_cnn_restnet_hf.py` recipe re-grounded
(same staging, new encoder): replay each of the 6,902 decisive games move-by-move through
`hexo_engine`; at each position call the **hexline Rust sample encoder** (the same
`encode_compact_row` self-play uses — one source of truth for window/legal/history facts) and
write `geo_compact` shards with a one-hot policy on the human move and hard-z value from the
engine-verified winner; prefit with the production trainer step; save
`{model_state, optimizer_state, epoch: 0}` and strict-reload-verify. ≈431k positions.

Gates: (a) legal-masked policy CE on a held-out split ≤ the dense_cnn_restnet prefit reference at
matched passes (the legal-CE numbers are comparable across lineages by construction — both are CE
over the same engine legal sets); (b) value sign-accuracy > 0.62 on held-out late-game positions;
(c) D6 probe: mean policy KL between augmented forwards < 0.05 nats after prefit.

### 8.2 Distillation from existing self-play shards (optional, cheap)

Existing dense/restnet compact shards store raw facts including **full placement history** —
sufficient to rebuild the engine state exactly. Converter: replay history through `hexo_engine`,
call `encode_compact_row` for native facts (windows recomputed by the engine, not approximated),
copy policy / opp_policy / value / stvalue / moves_left targets row-for-row. This yields a hexline
replay window seeded with real search targets (stronger than BC) without running a single self-play
game. Used, if at all, as a second prefit stage before the first self-play epoch; not load-bearing.

---

## 9. Code architecture

### 9.1 Package layout (greenfield)

```
packages/hexline/
  pyproject.toml                      # entry point: [project.entry-points."hexo_train.models"] hexline = "hexline.plugin"
  python/hexline/
    constants.py      # F=17, C=96, VALUE_BINS=65, MOVES_LEFT_CAP=512, bucket ladder, plane indices
    geometry.py       # DIR, packed keys, hex_distance, support/halo build, BFS dist, nbr tables
    d6.py             # the 12 axial transforms + σ_s axis table (self-contained ~60 lines)
    features.py       # node feature construction from raw facts (numpy)
    samples.py        # game finalization: targets, STV EMA, surprise weights, truncation drop
    compact_io.py     # geo_compact v1 reader/writer
    architecture.py   # HexConv7, ConvBlock, RelBiasAttention(sdpa|materialized), HexlineNetwork
    losses.py         # segment CE, binned value loss, hexline_loss (weights per §4)
    collate.py        # batch dict assembly + L-bucket grouping helpers (shared by trainer & inference)
    inference.py      # HexlineInference: payload ABI, grouping, priors/values (fp16 model path)
    replay.py         # KataGo window/shuffle over geo_compact
    trainer.py        # micro-batched train step, AdamW groups, AMP, checkpoints
    selfplay.py       # continuous-scheduler driver (Python side), .hxr + shard writes
    player.py / evaluation.py / plugin.py / checkpoints.py / rust_bridge.py / debug_artifacts.py
  rust/src/
    lib.rs            # module registration (hexo_models._rust.hexline)
    encoding.rs       # support set, features, payload assembly (rayon, f16 via half)
    sample_gen.rs     # compact-row facts for selfplay + bootstrap (encode_compact_row)
    mcts.rs           # session, lockstep + continuous scheduler, PCR/policy-init, seed streams
    mcts_tree.rs      # PUCT tree, widening, FPU, virtual loss, noise, reuse, TSS hook
    mcts_eval.rs      # cache, dedup, payload call, finalize_priors
    state.rs          # py-state cloning bridge
```

### 9.2 Build & shared-infra linkage

hexline's Rust compiles as a `#[path]`-included module of the **existing host crate**
`hexo_models` (`hexo_models/rust/src/lib.rs:1-40` — "the ONE Cargo crate / ONE maturin cdylib that
physically hosts every lineage's native accelerator"; hexgnn, a sibling *package*, is included
exactly this way). This is a 3-line additive edit to `lib.rs` (declare
`#[path] mod hexline; register submodule`) — flagged as the **only shared-file touch** — and buys:
`crate::threats_shared` linkage without visibility changes, the established maturin/WSL build
script flow (`scripts/_rebuild_hexo_models_hexgt.sh` pattern), and `hexo_engine`/`hexo_utils`
workspace deps for free. Python submodule: `hexo_models._rust.hexline`; all hexline Python reaches
it only through `hexline/rust_bridge.py`.

Shared infra linked, never forked: `hexo_engine` (state/legal/windows/apply), `hexo_utils`
(`hash_state`, .hxr records, sample buffers), `hexo_train` (pipeline/config/diagnostics/artifacts;
plugin per `registry.py:37-58` + the wider duck-typed hooks noted at `registry.py:20-24`:
`generate_selfplay`, `evaluate_epoch`, `select_training_samples`, `train_passes`, `close`),
`threats_shared`. Nothing is imported from dense_cnn/restnet/hexgt/hexgnn Python or Rust.

### 9.3 Parity / oracle test strategy (the correctness lattice)

1. **Geometry property tests**: halo == distance-9 ring (post-ply-0); support connectivity;
   DIR rotation orbit; σ_s table vs basis-vector transforms (cross-checked against `d6.py`).
2. **Conv equivalence oracle**: gather-conv == dense_cnn `HexConv2d` on a full-disk support with
   tap-mapped weights (fp64 bit-compare). Anchors the new primitive to the trusted one.
3. **Attention oracle**: sdpa path == materialized path (restnet's proven dual-impl gate), incl.
   padded groups and token blocks; fp16 mask-constant safety test (no −inf, no NaN rows).
4. **Encoder parity**: Rust `encoding.rs` vs Python `features.py` on replayed positions — identical
   node order, features (f32 pre-f16), nbr, legal_index.
5. **D6 consistency**: `expand(row, s)` == σ_s-permuted/coordinate-transformed `expand(row, 0)`
   (features, targets, masks), all 12 s.
6. **Payload contract**: byte-length validation, positional prior zip, finalize fail-loud cases
   (duplicate action, zero mass, terminal row) — mirror dense_cnn's tests.
7. **Search semantic parity**: scripted-evaluator A/B vs dense_cnn search (visits, chosen moves,
   noise/widening traces at fixed seeds); TSS on/off; cache hit/dedup accounting.
8. **End-to-end**: BC-prefit smoke + one CPU debug self-play game + shard→expand→train→checkpoint
   round-trip. Tests are authoritative in the WSL venv (project convention).

---

## 10. Perf budget

### 10.1 Parameters (exact, C=96, H=4, F=17, bins=65)

| component | count |
|---|---|
| stem (7·17·96 + BN 192) | 11,616 |
| 6 × conv block (129,408) | 776,448 |
| 3 × attention block (proj 37,248 + LN 384 + MLP 37,152 + table 948 = 75,732) | 227,196 |
| 8 summary tokens (8·96) | 768 |
| policy head (7·96·96+96 + 96+1) | 64,705 |
| opp_policy head | 64,705 |
| value MLP (192·96+96 + 96·65+65) | 24,833 |
| aux MLP + 4 tops (18,528 + 4·6,305) | 43,748 |
| **total** | **1,214,019** |

### 10.2 FLOPs per evaluation (MACs; L = support size; tokens add <1%)

`total ≈ 1.136M·L + 576·L²`  (stem 11.4k·L; convs 774k·L; attn proj+MLP 221k·L; scores+AV 576L²;
heads 129k·L).

| L | hexline MACs | vs dense_cnn_restnet (fixed ≈3.30G at any position) |
|---|---|---|
| 300 (early) | 0.39 G | 0.12× |
| 900 (mid) | 1.49 G | 0.45× |
| 1500 (late) | 3.00 G | 0.91× |
| 3000 (marathon) | 8.59 G | 2.6× — exactly the positions the crop could not represent at all |

The compute follows the true position size; the self-play average (length-weighted, most positions
mid-game) lands near 0.5× the dense baseline. GPU-batch utilization note: convs are
gather + one (M,7C)×(7C,C) GEMM — they batch across *nodes*, not rows, so small-row batches still
saturate (M ≈ rows·L̄ ≈ 50k node-rows at the measured avg batch 54).

### 10.3 Memory: the attention-bias budget (the one real cost of exact per-pair geometry)

Per A layer and micro-group: `bias = H·(Lp+8)²·2 B` per row; `relidx` (shared by all 3 layers)
`(G, Lp, Lp) i32`. Budget rule `G_max(Lp) = ⌊192 MiB / (8·(Lp+8)²)⌋`:

| Lp | G_max | bias/group | relidx |
|---|---|---|---|
| 512 | 93 | ≤192 MiB | ≤100 MiB |
| 1024 | 23 | ≤192 MiB | ≤100 MiB |
| 1536 | 10 | ≤183 MiB | ≤95 MiB |
| 2048 | 5 | ≤169 MiB | ≤88 MiB |
| 3072 | 2 | ≤145 MiB | ≤76 MiB |

- **Training (batch 32, AMP)**: weights+grads+Adam ≈ 19 MB; flat activations ≈ 0.3 GB; one
  micro-group's attention transients ≤ ~0.3 GB alive at a time (grad accumulation frees between
  groups; sdpa mem-efficient backward never materializes N²). Comfortable ≪ 12 GB shared budget.
- **Inference (256-leaf chunk, fp16 model)**: groups stream sequentially under the same budget;
  peak ≈ 0.4 GB transient. The eval cache and engine state stay in Rust as today.

### 10.4 fp16 / compile / TRT story

- **fp16**: f16 features on the wire (Rust `half` SIMD, the dense pattern); pure-fp16 model weights
  at inference with fp32 value decode + fp32 segment softmax for priors (restnet's gated recipe);
  the −3e4 mask constant is fp16-exact (§3.4).
- **torch.compile**: the forward is gathers + GEMMs + sdpa — compile-friendly, *no cuDNN convs*,
  so the 925 ms-per-novel-shape autotune hazard that forced dense bucketing mostly vanishes;
  buckets exist for compile-graph reuse instead: Lp static per bucket (≤9 graphs), batch dim marked
  dynamic. The data-dependent bias is computed in-graph from coords (no frozen-bias trick needed —
  there is no per-weight bias cache to freeze; the table gather is part of the graph).
- **TRT**: not promised. The model is GEMM/sdpa-dominated where TRT's edge over cuBLAS is small,
  and dynamic-shape gather/scatter export is exactly the brief's flagged risk. Story: fp16 +
  compile first; revisit TRT post-MVP only via per-bucket static-shape ONNX if profiling shows the
  projection GEMMs dominating (they shouldn't).

### 10.5 Encoder cost

Per state: support hash build (~1.6k inserts), halo expansion (6·N lookups), nbr (6·N), window scan
(≤ ~18·stones entries), BFS ≤ 9 rounds — tens of µs, rayon-parallel across the chunk
(`mcts_eval.rs:247-251` pattern). Wire size per state at L=1500: 17·2·1500 (feat) + 4·1500 (coords)
+ 24·1500 (nbr) ≈ 93 kB vs dense's fixed 1681·13·2 ≈ 44 kB — about 2× at the late-game tail but
*smaller* than dense below L≈700, and the zero-copy buffer path makes the crossing cheap either
way (nbr is 40% of it; if profiling shows the copy mattering, nbr can be derived on-GPU from
coords instead — kept on the wire initially for simplicity and Rust/Python parity testing).

---

## 11. Milestones (acceptance-gated; no GPU-scheduling decisions)

- **M0 — geometry kernel (Python).** `geometry.py`, `d6.py`, `features.py` + property tests (§9.3.1).
  Gate: all geometry/σ tests green.
- **M1 — architecture.** `architecture.py` with dual attention impls. Gates: conv-equivalence
  oracle (§9.3.2) bit-exact; sdpa==materialized ≤1e-6 rel; param count == 1,214,019;
  fp16 mask-safety test.
- **M2 — data plane.** `compact_io.py`, `samples.py`, `collate.py`, `losses.py`, golden-row tests,
  D6 consistency (§9.3.5). Gate: expand(s) equivalence for all 12 s; loss fail-loud cases covered.
- **M3 — BC prefit.** Bootstrap script + trainer. Gates: §8.1 (a)–(c). (This is also the cheap
  empirical check on the line-feature refinement: an A/B prefit with line channels zeroed must not
  beat the full model — if it does, the refinement is reverted to the binary-hot port, a 1-line
  feature-table change.)
- **M4 — Rust encoder + payload.** `encoding.rs`, `sample_gen.rs`, payload ABI. Gates: encoder
  parity (§9.3.4); payload byte-contract suite; `HexlineInference` round-trip on replayed states.
- **M5 — search.** `mcts_tree.rs`, `mcts.rs`, `mcts_eval.rs`. Gates: scripted-evaluator semantic
  parity vs dense_cnn (§9.3.7); TSS toggle test; cache/dedup accounting; deterministic seed-stream
  replay.
- **M6 — pipeline integration.** Plugin, selfplay driver, replay/shuffle, checkpoints, sidecar/
  dashboard tags, CPU debug-worker game. Gate: full epoch loop (selfplay→shard→shuffle→train→
  checkpoint→eval hook) on a short GPU smoke (active_games 32, visits 64).
- **M7 — perf.** fp16 model, bucket ladder + budget chunking, torch.compile per-bucket. Gate:
  ≥0.8× dense_cnn evals/s on a mid-game L-mix benchmark at the production visit budget; VRAM
  ceiling test alongside a concurrent training step.
- **M8 — strength & extras (optional).** Distill converter (§8.2); anchored-ladder arena vs
  dense_cnn reference checkpoints (shared eval-stage semantics); frontend/debug artifact parity.

Primary open risks tracked against gates: BN under micro-group statistics (M3/M6; LN fallback),
bias-budget throughput at the L≥2048 tail (M7; budget knob), search-rewrite fidelity (M5 harness),
line-feature distribution shift (M3 A/B).

---

## 12. Envelope deviations (everything changed vs §3 defaults; no §2 contradictions found)

1. **`own_hot`/`opp_hot` (2 binary planes) → 6 graded, axis-typed line-potential channels**
   (§2.2/2.3). Geometry-forced: the win condition is per-axis; D6 acts on the channels by an exact
   permutation (§5); information strictly contains the trusted planes. Empirically gated at M3 with
   a pre-committed reversion path.
2. **Dropped the `placements_made < 7` hot gate** (came with #1; graded counts self-attenuate).
3. **Rel-pos far field: distance rings 9–16 split into on-win-axis / off-win-axis classes**
   (+8 rows/head → table 237×4 per layer). The split predicate is D6-invariant; pure rings erase
   the lattice's anisotropy exactly where window-chains live (§3.4).
4. **Attention mask constant −3.0e4 instead of −1e9** — fp16-exact; closes the documented restnet
   fp16 saturation footgun (`architecture.py:89-97`).
5. **Shard dtype hygiene**: `phase` stored u8 (not object); history rows store (coord, owner,
   placement_index) only.
6. **Input channels 17 (not 13)** — consequence of #1 (net +4).
7. **Truncated games: rows dropped at write** — adopting main_4's C2 (`drop_truncated_rows`);
   the design then honors "hard-z" with z ∈ {−1, +1} only.
8. **opp_policy softmax support = current legal nodes** (dense used the whole 1681-cell crop);
   out-of-support target mass dropped + renormalized, with sidecar telemetry (§4.2). Forced by the
   owner-locked "halo carries no logits" + per-node vocabulary.
9. **Rust packaging**: hexline's greenfield Rust is `#[path]`-included into the `hexo_models` host
   crate (the hexgnn precedent) rather than a standalone cdylib — a 3-line additive edit to the
   shared `lib.rs`, flagged loudly here, in exchange for `threats_shared` linkage and the proven
   build flow.
10. Everything else in §3 is kept as specified: interleave `C C C A C C A C A`; token count 8 and
    2/2/4 split; channels 96 / 4 heads / mlp_ratio 2; BN-in-conv-blocks with LN fallback knob;
    217 exact bias entries within radius 8; 3 token bias scalars/head; sdpa + materialized oracle;
    ~1.2M params (1.214M exact).
