# hexion — a stone-anchored native-hex lineage designed backward from gradient flow

Design competition entry. Lens: **trainability and stability**. Every discretionary choice below is
subordinated to five questions: is the target exact, is augmentation exact, is train==eval exact, can one
sample's geometry perturb another sample's gradients, and will tomorrow's collapse be visible in tonight's
diagnostics. Envelope: `scripts/_wf_nm_brief.md` (owner-locked §2 honored in full; §3 refinements listed
loudly in §12 of this doc).

---

## 1. Identity

**hexion** is a greenfield Python+Rust lineage whose model is a set network over the stone-anchored support
set (stones ∪ full engine legal set ∪ 1-ring halo), with direction-typed 7-tap hex convolutions for local
reasoning and rel-pos-biased attention with 8 bidirectional summary tokens for global reasoning, in a
C C C A C C A C A trunk at 96 channels (~1.20 M params). The training-first thesis: **(a)** all
normalization is per-node LayerNorm in a pre-norm residual stream — zero batch-statistics state, exact
train/eval parity, exact micro-batch equivalence under variable-N batching, no running-stats shock at the
BC→RL transition; **(b)** policy logits exist *only* at legal nodes, so legality masking is structural
rather than a loss-time patch; **(c)** the main-value pathway is isolated from auxiliary heads at the
*token* level (the heads_v3 lesson applied natively); **(d)** D6 augmentation is exact for every row with
zero drops, because the support set commutes with the symmetry group (no crop boundary exists to clip
facts); **(e)** a fixed-probe drift-telemetry harness plus a per-head gradient panel makes policy churn,
value mis-calibration, and aux interference visible per epoch — the instruments main_1/main_3 forensics
had to reconstruct after the fact. Search semantics (batched PUCT, PCR, policy-init, continuous scheduler,
TSS toggle, eval cache) are ported semantically and reimplemented from scratch in a new crate; the training
loop reuses the proven dense_cnn/restnet pipeline semantics (KataGo window, policy-surprise rows, spawn-pool
expansion, AdamW/AMP/clip) re-expressed over the new representation.

---

## 2. Input representation

### 2.1 Support set

For engine state `s` with stone set `St`, engine legal set `L` (empty cells with hex distance ≤ 8 of any
stone; `LEGAL_RADIUS=8`), define

```
support(s) = St ∪ L ∪ halo,   halo = { c ∉ St∪L : c adjacent to some cell of St∪L }
```

Hex distance: `d((q1,r1),(q2,r2)) = max(|dq|, |dr|, |dq+dr|)` with `dq=q1−q2, dr=r1−r2`.

**Halo identity (provable, used as a test invariant):** every cell with `d_nearest(c) ≤ 8` is in `St∪L`
(occupied → stone; empty → legal), and every cell at `d_nearest = 9` has a neighbor at `d_nearest = 8`
(shorten a shortest path by one step). Hence `halo = { c : d_nearest(c) = 9 }` exactly — the closed
1-ring shell. Halo cells carry features but **never logits** (owner-locked §2.1).

**Node order (canonical):** ascending lexicographic `(q, r)`. Both featurizers (Rust serve-time, Python
train-time) MUST emit this order; a byte-equality parity test enforces it (§9). Coords stored `i16`.

**Ply-0 / empty board:** `St = ∅`, `L = {(0,0)}` (forced origin opening; engine truth, brief §1).
`support = {(0,0)} ∪ 6 neighbors`, N = 7. `d_nearest` is undefined with no stones → the distance feature
is defined as 1.0 everywhere (see table). All recency/hot/last-turn features are 0. The position is
trivially playable (1 legal move) and trains as a normal row (the BC corpus contains it).

**Scale:** N ≈ 271 at 1 stone (217-cell disk + 54-cell halo), ≈ 600–1500 mid-game, up to ~3k for long
spread games (brief §4). **No architectural cap** — no crop, ever (owner-locked). Buckets (§7.3) scale to
any N; a diagnostic warning fires at N > 4096 and the collate falls back to micro-buckets of 1 row.

### 2.2 Node feature table (13 features, port of the trusted planes)

`x ∈ R^{N×13}`, f32 at train expansion, f16 transport at the eval boundary (all features bounded [0,1] —
loss-free in f16, same argument as `mcts_eval.rs:270-276`). `P` = side to move; `latest = max placement
index`; reference semantics: `dense_cnn_restnet/input.py:55-157`, `constants.py:29-41`.

| # | name | formula (per node v) | notes |
|---|------|----------------------|-------|
| 0 | own_stone | 1 if v ∈ St owned by P | |
| 1 | opp_stone | 1 if v ∈ St owned by opponent | |
| 2 | empty | 1 − f0 − f1 | 1 on legal and halo |
| 3 | legal | 1 if v ∈ L | defines the logit set |
| 4 | phase_second | 1 if phase == SecondStone (const) | Opening/FirstStone → 0 |
| 5 | first_stone_of_turn | 1 at this turn's first placement cell (SecondStone only) | one-hot cell |
| 6 | player_colour | 1 if P == player0 (const) | |
| 7 | own_recency | max over own placements at v of `1/(1 + latest − idx)` | 0 off own stones; max-per-cell mirrors input.py:126-137 |
| 8 | opp_recency | same for opponent | |
| 9 | own_hot | 1 if v ∈ own active ≥count-3 window cell set | engine window facts (fact lists authored Rust-side, as today) |
| 10 | opp_hot | 1 if v ∈ opponent's ≥count-3 window cell set | |
| 11 | dist_nearest_stone | `min(d_nearest(v), 8) / 8` | replaces crop-center distance; halo → 1.0; empty board → 1.0 everywhere |
| 12 | opp_last_turn | 1 at cells of opponent's last full turn (≤2) | |

Every feature is a D6-invariant scalar attached to a coordinate; only coordinates transform under
augmentation (§5). Constant features (4, 6) are broadcast per node deliberately (keeps the encoder a single
N×13 tensor; cost negligible).

### 2.3 Auxiliary per-row arrays produced at featurize time

| array | shape/dtype | content |
|---|---|---|
| `coords` | (N,2) i16 | axial (q,r), canonical order |
| `nbr_idx` | (N,7) i32 | tap 0 = self; taps 1..6 = neighbor index in direction DIR[d]; missing (outside support) → sentinel (§3.2) |
| `legal_idx` | (Lg,) i32 | node indices of legal cells, ascending |
| `legal_action_ids` | (Lg,) u32 | packed engine ids aligned 1:1 with `legal_idx` (Rust keeps these; Python training uses them for target projection) |

Direction order `DIR = [(1,0),(0,1),(−1,1),(−1,0),(0,−1),(1,−1)]` — chosen so rot60 (d6.py:129-131:
`(q,r)→(−r,q+r)`) maps `DIR[k] → DIR[(k+1) mod 6]` and the reflection `(q,r)→(q,−q−r)` maps
`DIR[k] → DIR[5−k]`. D6 conjugation of adjacency is pure index arithmetic — the augmentation test (§5)
exploits this.

---

## 3. Trunk

State per row: `cells (N_pad, C=96)`, `tokens (8, C)`. Tokens are a learned `(8,96)` parameter
(trunc_normal 0.02), broadcast per row. Tokens are carried **untouched through C blocks** and participate
in **every A block** (owner-locked §2.6: join at the first attention layer = block 4, participate in all
attention layers).

### 3.1 Stem

`cells = x @ W_stem + b` — a per-node `Linear(13→96)`, no norm, no activation. Rationale: inputs are
bounded indicator/[0,1] features; the first block pre-norms anyway; a conv stem would duplicate block 1's
job. Receptive coverage is unaffected (radius 6 after C1-3, see 3.4). (§12 deviation D4.)

### 3.2 Conv block "C" (pre-norm, direction-typed, dense_cnn-family op)

```
h = LN(cells)                      # LayerNorm over channels, per node
u = HexConv7_a(h) ; u = ReLU(u)
u = HexConv7_b(u)                  # W_b zero-initialized (see 3.5)
cells = cells + u                  # tokens pass through unchanged
```

`HexConv7(h)_i = W_0 h_i + Σ_{d=1..6} W_d h_{nbr_d(i)} + b`, one 96×96 matrix per relative direction,
shared everywhere — **mathematically the same family as dense_cnn's hex conv** (7 active taps of the masked
3×3, `architecture.py:163-179`), with missing neighbors contributing zero (= conv zero-padding semantics,
§3 of the brief).

Implementation: append one all-zero row to `cells_ext = cat([h, zeros(1,C)])`; `nbr_idx` sentinel = that
row; gather `(N,7,C)` → reshape `(N,7C)` → single matmul with `W ∈ R^{7C×96}` (blocks `[W_0;W_1..W_6]`).
One gather + one GEMM per conv; padding-free in flat layout, padding-inert in padded layout (a real node's
`nbr_idx` never references a pad node, because adjacency is built only among real support cells — pad rows
compute garbage that no real row ever reads).

**Norm choice — LayerNorm, not BatchNorm (§12 deviation D1, the load-bearing one).** Under variable-N
flat/padded batches, BatchNorm statistics are means over *nodes*, so a single 3k-node marathon row
dominates the statistics of every other row in the batch — cross-sample gradient coupling keyed on game
length, exactly the failure axis this repo keeps fighting (crop-frozen marathons, length-decay knees).
BN also (i) makes train-mode and eval-mode different network functions (running stats), so training metrics
are measured on a different function than self-play uses — a diagnosis tax the value-head autopsy paid;
(ii) couples the BC→RL transition to running-stat re-adaptation lag; (iii) breaks exact micro-batch
equivalence (§7.3 depends on it); (iv) needs fold/fuse machinery for fp16 inference. LayerNorm has none of
these failure modes, costs O(C) per node (negligible vs the 7·96² conv GEMM), and the trunk's A blocks are
already LN. The brief's §3 names "LN as fallback knob"; hexion promotes the fallback to the only mode.

### 3.3 Attention block "A" (pre-norm transformer over [tokens ; cells])

```
z = concat([tokens, cells], dim=seq)            # (8+N_pad, 96)
z = z + MHSA(LN(z); additive bias B̂)            # 4 heads, head_dim 24, scale 1/√24, out_proj zero-init
z = z + MLP(LN(z))                              # Linear(96→192) → GELU → Linear(192→96), fc2 zero-init
tokens, cells = split(z)
```

`MHSA` follows the restnet pattern (`architecture.py:262-343`): q/k/v/out `Linear(96→96)`; two
numerically-identical paths — `sdpa` (default; bias as additive `attn_mask`) and `materialized` (the
correctness oracle, tested ≤1e-5 against sdpa).

**Rel-pos bias (one shared table for all A layers; §12 deviation D2).** A single learned table
`T ∈ R^{229×4}` (per head):

| index range | meaning | count |
|---|---|---|
| 0..216 | exact entry per axial offset with hex distance ≤ 8 | 217 |
| 217..224 | ring bucket per distance 9..16 | 8 |
| 225 | far bucket (distance ≥ 17) | 1 |
| 226 / 227 / 228 | token→cell / cell→token / token→token | 3 |

Index computation, **on device, once per forward**, from `coords`:
`dq = q_i − q_j`, `dr = r_i − r_j` (B,N,N) i32; `dist = max(|dq|,|dr|,|dq+dr|)`;
`idx = dist≤8 ? EXACT_LUT[(dq+8)·17+(dr+8)] : (dist≤16 ? 217+dist−9 : 225)` with `EXACT_LUT` a
precomputed 289-entry i32 buffer (offsets within the |dq|,|dr|≤8 square that are out-of-disk are never
queried — guarded by `dist≤8`). The full (B, 8+N, 8+N) index matrix gets borders filled with constants
226/227/228, then **one** `F.embedding(idx, T)` → permute → `B̂ (B, 4, 8+N, 8+N)`, plus the additive
key-padding mask (−1.0e9 on pad-cell KEY columns only). `B̂` is computed once and reused by all three A
layers (same table → same tensor → autograd saves one storage, not three).

*fp16 saturation note:* −1e9 saturates to −inf under fp16 — safe here for the same reason as restnet
(`architecture.py:88-97`): only KEY columns are masked and **every query row always has ≥8 un-masked token
keys**, so a fully-masked softmax row is structurally impossible (the tokens double as a NaN-row guarantee).

**Why one shared table:** with variable support sets the bias matrix is per-row (unlike restnet's fixed
geometry where one (1,4,1681,1681) gather is shared batch-wide, `architecture.py:319-329`). Per-layer
tables would mean three (B,4,S,S) gathers alive for backward; one shared table means one. Cost: 916 params
and one gather; expressivity loss is marginal at 3 A layers. `per_layer_bias=true` remains a config knob.

### 3.4 Interleave `C C C A C C A C A` (brief §3 default, kept) — per-layer rationale

| layer | kind | why here |
|---|---|---|
| 1–3 | C | local tactical features to receptive radius 6 ≥ the 5-span of a 6-window, *before* any attention — attention starts from window-aware features, not raw indicators |
| 4 | A | first global mix; token hub bootstrap (tokens read the board, board reads token init — round 1 of the bidirectional hub) |
| 5–6 | C | local refinement of globally-conditioned features (radius 10) |
| 7 | A | second token round — the hub is now functional (cell→token→cell global paths need ≥2 A layers) |
| 8 | C | final local sharpening for the policy head (radius 12) |
| 9 | A | trunk ends on A: value/aux tokens read the final board state; policy nodes get final global context (brief §3 rationale, kept) |

Final `LN` applied to both cells and tokens before all heads (standard pre-norm closure).

### 3.5 Initialization (identity-at-init; §12 deviation D3)

trunc_normal(std 0.02) on every weight matrix (incl. conv blocks-as-matmul and stem); biases 0; LN weight 1
bias 0; token inits trunc_normal 0.02; bias table zeros — **plus zero-init of the last matrix of every
residual branch** (`HexConv7_b`, attention `out_proj`, MLP `fc2`). Every residual block is the identity at
step 0; the whole net at init is stem → final LN → heads. Gradient flow at init is perfectly conditioned
regardless of depth; removes any depth-9 warmup fragility for free (no extra mechanism, just an init rule).

---

## 4. Heads & losses

All five heads, exact shapes. Per row: `E_cells (N,96)`, `E_tok (8,96)` post-final-LN. Loss reduction is
always **mean over rows** (never over nodes) — a 1500-legal row and a 200-legal row contribute equally,
matching dense-crop semantics; this is stated as a contract because node-mean weighting is the classic
variable-N bug that silently over-weights marathons.

| head | reads | architecture | output |
|---|---|---|---|
| policy | cells | `u = ReLU(HexConv7_p(E_cells))`; `logit_i = u_i @ w_p` (96→1); **gather at `legal_idx` only** | (Lg,) logits |
| opp_policy | cells | same structure, separate params | (Lg,) logits |
| value (MAIN) | tokens 0,1 | `MLP_v: Linear(192→96) → ReLU → Linear(96→65)` on `concat(E_tok[0],E_tok[1])` | (65,) logits |
| stvalue_h, h∈{2,6,16} | tokens 2,3 | shared `a = ReLU(Linear_aux(192→96))` on `concat(E_tok[2],E_tok[3])`; per-horizon `Linear(96→65)` | (65,) ×3 |
| moves_left | tokens 2,3 | `Linear(96→65)` on the same `a` | (65,) |

Tokens 4–7 are uncommitted hub capacity (brief §3 split 2/2/4, kept). **Main-value isolation:** the only
parameters shared between the main value head and anything else are the trunk itself — the heads_v3 lesson
(`architecture.py:14-20`) is enforced at the token level, stronger than two ValueReductions over one
feature map: aux targets cannot even compete for the main head's *readout features*, only for trunk
capacity (which is what aux heads are for).

### 4.1 Targets, masking, loss formulas

`CE(p_logits, t) = −Σ_k t_k · log_softmax(p_logits)_k`. 65-bin support `bins = linspace(−1,1,65)`;
scalar→binned: `pos = (z+1)·32`, mass split `floor/ceil` (adjacent-bin soft target, exactly
`losses.py:33-53`). Expected-value decode `E[v] = softmax·bins` (`losses.py:26-30`).

| head | target construction | masking | weight |
|---|---|---|---|
| policy | MCTS visit distribution over action ids → mapped to `legal_idx` positions, normalized | none needed: target support ⊆ legal by construction (validated, fail-loud) | 1.0 |
| opp_policy | next opponent decision's visit policy (`samples` finalize, `samples.py:330-354`, incl. the PCR `mask_from_fast` knob) **projected onto THIS position's legal set**: keep entries whose action ∈ this row's `legal_action_ids`, renormalize | row masked when (a) no future opponent decision, (b) masked-from-fast, or (c) projected mass = 0. Denominator = # unmasked rows in the optimizer step | 0.25 |
| value | hard z ∈ {−1,0,+1} from side-to-move perspective → binned (owner-locked hard-z). `soft_z_lambda` blend knob retained, default **0.0** | `drop_truncated_rows` knob (default false) mirrors restnet quarantine semantics (`config.py:122-128`) | 1.0 |
| stvalue_h | EMA of future root values with **even-offset (turn-aligned) stepping** and decay `(h−1)/(h+1)`, normalized by weight sum — exact port of `samples.py:357-402` (horizons must be even; enforced at config parse) | masked when no even-offset future decision exists; masked mean (denominator = unmasked rows; zero-denominator → exact 0 via the `logits.sum()*0.0` device-safe path, `losses.py:128-131`) | 0.1 each |
| moves_left | `ml` = decisions remaining after this one; scalar `2·min(1, ml/512) − 1` → binned. Cap = 512 applied at **expansion**, not baked into shards (`constants.py:14-27` owner directive preserved) | masked for truncated games / absent (−1 sentinel) | 0.1 |

`total = 1.0·policy + 1.0·value + 0.25·opp_policy + 0.1·Σ_h stv_h + 0.1·moves_left` (production weights,
`configs/dense_cnn_restnet_main_4.toml:127-134`).

**Structural legality masking:** because logits are *gathered at legal nodes*, the softmax support is the
legal set by construction — the new-lineage equivalent of restnet's masked CE (`losses.py:56-97`), with the
−1e9 fill and "no mass outside mask" check made unnecessary at the loss (the projection step still validates
target support, fail-loud). Serve-time and train-time supports are *identical objects*.

**Opp-policy coverage caveat (and why this is correct, not a hack):** the opponent's decision happens 1–2
placements later; new stones extend legality up to distance 8 around themselves, so a future-legal cell can
sit outside this position's support (up to distance 16 from current stones). Projection onto the current
legal set drops that mass — empirically tiny (the opponent answers locally), and *measured*: per-epoch
diagnostic `opp_target_coverage_mean` (kept mass fraction). restnet faced the same family of issue and chose
unmasked CE + allow-zero rows (`losses.py:149-159`); hexion chooses the masked projection because our head
has no logits off the legal set to absorb stray mass.

**Target hygiene (fail-loud at expansion, port of `losses.py` checks):** finite, non-negative, positive
mass where unmasked, support ⊆ legal for policy; value scalar ∈ [−1,1]. A violated row raises with shard
path + row index (no silent repair).

### 4.2 65-bin calibration notes (lens item)

With pure hard-z targets, only bins {0, 32, 64} ever receive target mass; CE then drives the predictive
distribution toward the conditional outcome distribution over those three points, so `E[v] = p_win − p_loss`
is calibrated by construction *if training is healthy* — calibration failures are therefore informative, and
hexion measures them per epoch (§7.5): ECE over E[v]-deciles vs realized z, `mean(E[v]) − mean(z)` (the
autopsy-endorsed optimism scalar), extreme-bin mass fraction, value-logit scale. The 21.6-nat relocation
shock (memory: value-head autopsy) is encoded as a procedure rule: **any mid-run change of value-target
family (e.g. enabling soft_z) requires a value-head LR ramp (0.1× → 1× over 2k steps) or a fresh head** —
written into the config docstring, because the failure was procedural, not architectural.

---

## 5. Symmetry — D6 training-time augmentation (owner-locked approach, made exact)

Group: 12 transforms about the **origin** (no crop center exists). Index 0–5 = rot60^k; 6–11 = reflect
then rotate. Maps (same integer math as `d6.py:129-136`): `rot60(q,r) = (−r, q+r)`;
`reflect(q,r) = (q, −q−r)`.

**Mechanics at expansion (per row, per epoch):** draw σ ~ Uniform[0..11] from a deterministic stream seeded
by `(run_seed, epoch, shard, row)` (the `trainer.py:545-559` recipe); transform the *coordinates* of every
fact — stones, legal action ids, placement history, hot cells, last-turn cells, first_stone, policy /
opp-policy action ids — then rebuild support set, canonical order, features, `nbr_idx`, targets from the
transformed facts. Nothing else changes: every node feature is a D6-invariant scalar (§2.2), and hex
distance is D6-invariant, so:

**Exactness lemma:** `support(σ·s) = σ·support(s)` and `featurize(σ·s) = P_σ · featurize(s)` where `P_σ` is
the node permutation induced by canonical re-sorting; adjacency conjugates with the direction relabeling
(`nbr_{ρσ(d)}(P_σ i) = P_σ nbr_d(i)`, with ρ_σ the index map from §2.3). **No fact can leave the
representable domain** — the disk-closure argument and the drop/spill machinery of the crop lineage
(`compact_io.py:293-322`) are deleted, not ported. Augmentation is exact for 100% of rows, all 12 σ.

**Bias table and conv weights are untouched** by augmentation — the table is indexed by transformed offsets
of transformed coordinate *pairs* (consistent), and the direction-typed weights see relabeled directions;
that relabeling is precisely what teaches approximate D6 equivariance (the trusted dense_cnn approach,
owner-locked §2.9 — no architectural invariance constraints).

**Validation uses identity symmetry** (deterministic val loss, restnet parity, `trainer.py:443-445`).

**Test suite (M0 gate):** (i) feature/adjacency/target equality under all 12 σ on ≥1k fuzzed positions
(integer-exact); (ii) composition table of the 12 maps is D6; (iii) σ∘σ⁻¹ = id via `inverse_index`
semantics; (iv) direction relabeling law `rot: d→d+1 mod 6`, `reflect: d→5−d` checked against the maps.

---

## 6. Search integration

### 6.1 Evaluator payload (Rust → Python), `hexion_eval_v1`

One dict per eval chunk (B rows, T total nodes, G total legal entries). Zero-copy buffers use the
PlaneBuffer pattern (`mcts_eval.rs:48-100`).

| key | type / dtype | layout |
|---|---|---|
| `schema` | str | `"hexion_eval_v1"` (version-gated, fail-loud on mismatch) |
| `num_rows` | int | B |
| `node_offsets` | tuple i64, len B+1 | CSR row offsets into nodes |
| `coords` | buffer i16, len 2·T | interleaved q,r — canonical per-row order |
| `features` | buffer f16, len 13·T | row-major (node, feature) |
| `legal_offsets` | tuple i64, len B+1 | CSR row offsets into legal entries |
| `legal_node_idx` | bytes u32, len G | per row: indices into that row's node slice, ascending |

Rust keeps the aligned per-row `Vec<PackedCoord>` action ids (they never cross the boundary — the hexgt
precedent: priors return positionally, `hexgt/rust/src/mcts_eval.rs:250-257`).

**Python evaluator:** group rows by node-count bucket (multiples of 256; preserves request order via an
index map), pad each group to `(B_g, N_pad)`, build `nbr_idx` on CPU per group (vectorized; or shipped from
Rust later if profiled hot — start CPU-Python, it is ~6 hash-free grid gathers), upcast features to f32
on-device, run `forward_policy_value`, per-row softmax over each row's legal logits by segmented logsumexp
(the `inference.py:385-402` scatter pattern), **clamp values to [−1,1]** before byte-out (the
`inference.py:366-372` lesson: a 1-ULP excursion must not abort a whole batched search).

**Return (identical contract to dense_cnn):** `{"values_bytes": f32×B, "priors_bytes": f32×G positional}`.
Rust zips positionally with its kept action ids, then validates finite/≥0/unique/positive-mass,
descending-sorts, normalizes to sum 1.0 → `RustEvaluation{value, priors}` — a semantic port of
`finalize_model_priors` (`mcts_eval.rs:515-580`). The PUCT tree consumes `(action_id, prior)` pairs
opaquely (model-agnostic, crop-free — brief §4), so the tree never knows the representation changed.

### 6.2 Cache

`HashMap<StateHash, Arc<RustEvaluation>>`, key = `hexo_utils::hash_state` (pure engine hash,
`hexo_utils/rust/src/state_hash.rs:31` — no encoder dependence), bounded ~1M entries, in-flight dedup of
duplicate misses — semantic port of `evaluate_model1_state_refs_cached` (`mcts_eval.rs:415-513`).

### 6.3 TSS toggle

Link `threats_shared` (§9 build story) and thread the `tss_enabled` config key (landed 2026-06-12) to the
three consumption sites (expansion injection, leaf override, root guard — `threats_shared.rs:12-16`).
Pleasant consequence of the full-legal vocabulary: "every tactical cell is always a legal move; the only
thing that can exclude one ... is the crop" (`threats_shared.rs:34-38`) — the crop call-site handling is
deleted; tactical cells are always in-vocabulary by construction.

### 6.4 PCR / policy-init / continuous scheduler — fresh implementation, exact semantic port

Decision: **reimplement in the new crate** (search is not on the blessed shared-infra list; brief hard rule
10), preserving these semantics value-for-value:

- `MoveClass{Full,Fast,Init}` with the PCR coin per `(base_seed, game_key, ply)` and policy-init draws —
  same `mix_seed` mixer and the six stream constants (`mcts.rs:60-66`), same truncated-exponential ply count
  (`mcts.rs:108-131`), same classify ordering (`mcts.rs:134-155`). **Fixture parity test:** capture
  (seed, game_key, ply) → (class, unit draws) tables from the existing implementation and assert
  value-equality in the new one — reproducibility semantics locked cheaply.
- Continuous scheduler: per-game `ContinuousSlot` (staggered roots, in-flight counts, flush decisions,
  baseline visit maps, policy_init_remaining — `mcts.rs:209-237`), plus the lockstep `search` entry point.
- Full search semantics: batched PUCT, prior-sorted lazy edge materialization, nucleus widening
  (policy_mass 0.95 / max_children 96 / min_children 2 — brief-locked numbers), FPU reduction +
  root-FPU-zero-under-noise knob, virtual loss, Dirichlet root noise (fraction/total_alpha), root policy
  temperature + early ramp, forced playouts (Full moves only), tree/subtree reuse, deterministic per-stream
  seeding. PUCT tie-break = higher prior, then lower action id (matches the dense_cnn sort).

Self-play emits per-decision compact samples + `.hxr` records via `hexo_runner` (shared), and the root
prior / visit policies as byte-backed compact policies (the `CompactVisitPolicy` memory discipline,
`samples.py:124-131`).

---

## 7. Data pipeline

### 7.1 Shard schema — `hexion_compact_v1` (compact-facts concept reused, new schema)

One `.npz` per game + JSON sidecar, columnar, RAW FACTS only (representation-agnostic; encoders expand at
train read — the proven two-lineages-one-shard property, brief §4). Layout family = `compact_io.py:56-226`
minus crop fields, plus hygiene fixes:

Per-row scalars: `turn_index` i32 · `current_player` u8 · `phase` u8 (enum {0:Opening, 1:FirstStone,
2:SecondStone} — replaces the object-dtype string column) · `value` f32 · `moves_left` f32 (−1 = masked) ·
`first_present` u8, `first_q`/`first_r` i16 · `stvalue` (n,h) f32 + `stvalue_mask` (n,h) f32.
CSR variable-length: `stones_qr` i16 + `stones_owner` u8 + offsets · `legal_ids` u32 + offsets ·
`hist_qr` i16 + `hist_owner` u8 + `hist_idx` i32 + offsets · `own_hot_qr`/`opp_hot_qr`/`last_hot_qr` i16 +
offsets · `pol_act` u32 + `pol_w` f32 + offsets · `opp_act` u32 + `opp_w` f32 + offsets.
Header: `schema_version` (=1), `num_rows`, `horizons` i32.

Deletions mirrored from restnet: `root_prior_policy`, `policy_surprise`, `frequency_weight` are dropped at
write (surprise weighting is baked into row duplication pre-write; `compact_io.py:8-16`). **No `center`
columns** — there is no crop. Sidecar JSON keeps the exact restnet field set (`num_rows`, `raw_rows`,
`effective_rows`, `epoch`, `game_id`, `target_schema_version`, `policy_surprise_mean`,
`frequency_weight_mean`, `created_at`) so the dashboard's sidecar parser keeps working unmodified.

### 7.2 Expand-time featurization (Python, worker pool)

Per row: apply σ (§5) → build support via a **bounding-box raster grid** (int32 grid over the support's
coord bbox; ~O(area) alloc, tens of KB): mark stones, mark legal, compute the d=9 halo shell by 6-direction
dilation of the support mask; node list = canonical sort of marked cells; neighbor lookup = 6 shifted grid
gathers (fully vectorized numpy — no per-cell Python loops, unlike `input.py`'s loops); features per §2.2;
targets per §4.1. Output arrays per row: `coords`, `feats`, `nbr_idx`, `legal_idx`, `policy (Lg,)`,
`opp_policy (Lg,) + opp_mask`, `value`, `stvalue_h + masks`, `moves_left + mask`.

Parity: the Rust serve-time featurizer implements the identical spec; M2's byte-equality test on fuzzed
states is the train/serve-skew killer.

### 7.3 Collate and variable-N batching (training)

A training batch = 32 rows (production batch size, ToML-tunable). Collate:

1. Sort the 32 rows by N; partition into **micro-buckets** where all rows share
   `N_pad = max(256, ceil(N_max_in_bucket/128)·128)`, subject to the bias-memory rule
   `B_g · N_pad² ≤ 2.0e7` (caps the transient (B_g,4,S,S) fp16 bias at ~160 MB).
2. Per micro-bucket: pad `feats/coords/nbr_idx/legal_*` to `(B_g, N_pad, …)`; `valid (B_g,N_pad)` bool;
   `nbr` pad slots → zero-row sentinel; legal pads masked.
3. One optimizer step per 32-row batch via gradient accumulation over micro-buckets, with **step-global
   denominators** (row count 32 for unmasked heads; per-head unmasked-row counts computed at collate for
   masked heads) so `loss_step = Σ_buckets Σ_rows ℓ / denom_step` — **mathematically identical to a single
   monolithic padded batch.** This identity holds because LayerNorm has no cross-row state (it would be
   false under BatchNorm) and is asserted by a unit test (§9), not assumed.

Padding waste under sorted micro-buckets is small (rows in a bucket are length-adjacent); no row's geometry
can influence another row's forward (padding-inertness is the M1 oracle test).

### 7.4 Replay / shuffle / trainer loop

Semantic ports, fresh code, same knobs and defaults:

- **Row weighting at finalize:** KataGo policy-surprise frequency weights (uniform_fraction 0.5, max_weight
  8.0, weights sum to game length pre-clamp; floor+Bernoulli duplication) + the dormant length-decay knees
  (moves-left knee/halflife, game-length knee/halflife) — exact semantics of
  `replay.py:178-304` including the truncated-row `ml` fallback.
- **Window build:** mtime-ordered scan, taper window (`compute_katago_window_rows`, exponent 0.65,
  expand_per_row 0.4, scale 50k), keep_target_rows 300k (production main_4), keep-prob subsample, md5
  validation split, permute, batch-aligned compact output shards + train/val/shuffle JSONs
  (`replay.py:346-548` semantics; mtime ordering documented → window-seeding tooling must `cp -p`).
- **Trainer:** duck-typed hooks `select_training_samples` / `train_passes` / `close` driven by the shared
  pipeline (`hexo_train/pipeline.py:54-109`, `registry.py:38-58`); train-bucket pacing
  (max_train_bucket_per_new_data 8, bucket cap, no_repeat_files) — `trainer.py:124-254,481-515` semantics;
  persistent spawn-pool shard expansion with the `_PARALLEL_MIN_ROWS` small-workload bypass
  (`trainer.py:89-101,584`).
- **Optimizer step:** AdamW lr 1e-3, wd 1e-4 on **matrix weights only** (no-decay set: all biases, LN
  params, token inits, bias table), betas torch-default; AMP autocast + GradScaler; unscale → global
  grad-clip 1.0 → step (`trainer.py:393-423` shape); **linear LR warmup over the first 500 optimizer steps
  of a fresh initialization only** (resumes skip it) — Adam second-moment burn-in insurance, §12 D5.
- **Validation:** identity symmetry, eval-mode — which under this design is *the same function* as train
  mode (LN, no dropout); asserted by a parity test. Val loss is measured on exactly the network self-play
  serves.

### 7.5 Per-epoch diagnostics contract (lens deliverable — exact keys)

Emitted into the epoch result (and thus diagnostics JSON) by `train_passes` / `evaluate_epoch`:

**Loss panel (every step, epoch-averaged):** `loss_components.{policy,value,opp_policy,stvalue_h,moves_left,total}`
(unweighted per-head CE + weighted total — restnet's legible split, `trainer.py:378-388`), plus
`opp_target_coverage_mean`, `masked_fraction.{opp,stv_h,moves_left}`.

**Optimization panel:** `grad_norm_preclip.{mean,p95,max}`, `clip_fraction`, `update_to_weight_ratio`
(‖Δθ‖/‖θ‖ per epoch), `amp_scale`, `nan_trips` (must be 0; a non-finite loss component dumps batch
provenance — shard paths + row indices — to the run dir and raises).

**Value calibration panel (validation pass):** `value_ece` (E[v]-decile reliability vs realized z),
`value_optimism` = mean(E[v]) − mean(z), `value_extreme_mass` (mean predicted mass on bins {0,64}),
`value_logit_scale` (mean max-logit), CE split by outcome class and by game-length quartile (the
length-coupled-poison axis from main_3, made visible), `moves_left_mae_decisions` (decode E[bins]→ml).

**Gradient-interference panel (once per epoch, one probe micro-bucket, 5 single-head backwards):**
`head_grad_norm.{policy,value,opp,stv,ml}` on trunk params; `grad_cos.policy_value`, `grad_cos.value_aux`.
Direct measurement of the interference the token split is designed to bound.

**Fixed-probe drift telemetry (novel):** at run start, freeze 1024 rows (with realized outcomes) into
`<run>/probe/probe_rows.npz`; every epoch forward them (identity symmetry, eval mode, seconds of GPU) and
persist outputs. Report `probe_policy_kl_prev` (mean KL(πₑ‖πₑ₋₁) over rows — a policy-churn rate meter; a
collapse/divergence ignition shows here epochs before Elo does), `probe_policy_entropy`,
`probe_value_shift_mean` (|ΔE[v]|), `probe_value_ece` (vs the frozen rows' real outcomes — longitudinally
comparable, unlike the drifting self-play CE floor), `probe_attention_token_mass` (mean attention mass
cells→tokens per A layer — token-hub health), `probe_attention_entropy` per layer.

**Data panel:** legal-set size distribution {p50,p90,max}, support-N distribution, rows with N>4096,
shuffle/window stats (inherited fields).

---

## 8. Bootstrap

**Phase A — BC prefit from the HF corpus** (timmyburn/hexo-bootstrap-corpus; 6,902 decisive games ≈ 431k
positions; raw move-lists). Fresh script in the new package, same plan as the proven
`scripts/bootstrap_dense_cnn_restnet_hf.py:1-35`:

1. CONVERT: replay each game through `hexo_engine`; per position author facts via the hexion Rust
   fact-builder; policy target = one-hot on the played move (root prior = same one-hot ⇒ policy-surprise
   KL = 0 ⇒ all frequency weights = 1.0, no pathological duplication — `replay.py:217-227` arithmetic);
   finalize with hard z (winner = last mover, cross-checked against engine terminal); write production
   shards with the production writer. Faithfulness gates per game: legality, terminal, decisive.
2. PREFIT: production trainer step; batch 64 (WSL prefit envelope: 64 fits, 128 OOMs), grad-accum to an
   effective 128 if useful; ~4 passes over the shuffled corpus; D6 augmentation ON (it is exact here);
   lr 1e-3 with the 500-step warmup; 5% md5 validation split.
3. SAVE `{model_state, optimizer_state, train_state, epoch:0}` + strict-reload verification.

Acceptance gates: held-out top-1 within 2 pts of the restnet prefit reference measured under the same
split; `value_ece ≤ 0.08` and monotone reliability; probe telemetry online (the probe set is drawn from the
BC val split at this stage); all M-series oracle tests green.

**Phase B — optional distillation from existing self-play shards.** restnet's compact shards store raw
facts, and our expander can read them through a small schema adapter (their `center` ignored; stones /
legal / policy / value / stv / moves_left all present). **Taint disclosed:** those rows' stored legal sets
and visit policies are crop-restricted at the source (`samples.py:295-298`), so the support sets built from
them inherit the radius-20 truncation. Acceptable for a warm-start (targets never claimed crop-free); rows
carry `source=legacy_shard` metadata; 1–2 passes over ~300k recent main_4 rows. **Not** used to seed the RL
replay window — the window starts from fresh hexion self-play only, so the on-policy buffer is never mixed
with crop-tainted rows.

**RL start:** carry prefit optimizer state; standard run config; the BC→RL transition needs no
normalization-statistics adaptation (LN) — the expected discontinuities are target-family ones (one-hot →
visit distributions; human → self-play outcomes), both visible on the probe panel from epoch 1.

---

## 9. Code architecture

```
packages/hexion/
  pyproject.toml                  # maturin build; entry point [hexo_train.models] hexion = hexion.plugin
  python/hexion/
    constants.py   geometry.py   d6.py        features.py   samples.py    shards.py
    replay.py      architecture.py  losses.py  trainer.py    inference.py  selfplay.py
    evaluation.py  player.py     plugin.py    config.py     rust_bridge.py probe.py
  rust/  (crate hexion-rust → module hexion._rust)
    src/lib.rs  src/constants.rs  src/facts.rs        # state → fact lists (stones/legal/hot/recency/last)
    src/featurize.rs                                   # facts → coords/features/legal payload arrays
    src/eval_bridge.rs                                 # payload assembly + return parsing + cache (§6)
    src/tree.rs  src/scheduler.rs                      # PUCT tree; lockstep + continuous, PCR, policy-init
    src/sample_gen.rs                                  # per-decision compact-sample facts
  tests/ (package-local) + tests/test_hexion_*.py (repo suite)
```

**Links against (blessed shared infra only):** `hexo_engine` (Rust dep + PyO3 types), `hexo_utils`
(`hash_state`), `hexo_train` (pipeline/registry/config/diagnostics — plugin mode 1, explicit
`[model].module = "hexion.plugin"`), `hexo_runner` (.hxr records, sealbot eval adapters), and
**threats_shared**: primary plan = promote `packages/hexo_models/rust/src/threats_shared.rs` to a tiny rlib
crate `hexo_threats` consumed by both `hexo_models` and `hexion-rust` via path dependency (a small,
owner-visible refactor; the file is self-contained over `hexo_engine`); fallback = `#[path]` include of the
same file with a cross-crate drift parity test. **Own cdylib** — deliberately *not* a fourth `#[path]`
submodule of `hexo_models._rust`, whose single-crate design makes every rebuild change search semantics for
all lineages at once (`hexo_models/rust/src/lib.rs:3-17`); greenfield must not enlarge that blast radius.

**No code copied** from dense_cnn/restnet/hexgt/hexgnn (brief hard rule 10): semantics are ported from this
doc's specs and the cited reference lines, then locked by fixtures.

**Oracle / parity test strategy (the lens's enforcement arm):**

| test | asserts |
|---|---|
| sdpa vs materialized attention | ≤1e-5, all scopes, with tokens + padding |
| padded-batch vs single-row forward | a row's outputs identical alone vs inside any micro-bucket (THE variable-N oracle) |
| micro-bucket grad accumulation vs monolithic batch | gradient equality (LN exactness claim, §7.3) |
| train-mode vs eval-mode forward | bit-identical (no dropout, no BN — parity is a *theorem* here, test keeps it true) |
| Python vs Rust featurizer | coords/features/legal byte-equal on ≥1k fuzzed engine states |
| D6 suite | §5 items (i)–(iv), integer-exact |
| halo shell | `halo == {d_nearest = 9}` on fuzzed states |
| mix_seed / PCR / policy-init | value-equality vs fixtures captured from the dense_cnn implementation |
| evaluator round-trip | random-weights model through the full Rust payload path == direct Python forward; malformed payloads raise |
| loss masking | empty-opp rows, masked stv/ml, zero-denominator exact-0 path, target-hygiene raises |
| checkpoint strict load | bidirectional key-set equality; mismatch raises (the "silent random value head" lesson, memory: main1 ep≤23) |
| shard round-trip | write→read→expand identity; schema-version gate |

**Build story:** maturin per-package build (mirrors hexo_models' pattern, own module name `hexion._rust`);
WSL venv canonical; `cargo test` covers tree/scheduler invariants without the python feature.

---

## 10. Perf budget

### 10.1 Parameters (exact, C=96, 4 heads, mlp_ratio 2)

| component | formula | params |
|---|---|---|
| stem | 13·96+96 | 1,344 |
| 6 × C block | 6 × (LN 192 + 2×(7·96²+96)) | 776,448 |
| 3 × A block | 3 × (2 LN 384 + QKVO 4·(96²+96) + MLP 18,624+18,528) | 224,352 |
| rel-pos table (shared) | 229×4 | 916 |
| tokens | 8×96 | 768 |
| final LN | 192 | 192 |
| policy + opp heads | 2 × (7·96²+96 + 97) | 129,410 |
| main value MLP | 192·96+96 + 96·65+65 | 24,833 |
| aux (shared 192→96 + 4 tops 96→65) | 18,528 + 4×6,305 | 43,748 |
| **total** | | **1,202,011 ≈ 1.20 M** |

### 10.2 FLOPs per eval (2 × MACs), vs dense_cnn_restnet's fixed-1681 trunk

Per-node fixed cost ≈ 1.13 M MACs (6 C blocks 774k + A linears 221k + heads 130k + stem ~1k); attention
pairwise adds ≈ `3 layers × 2·(8+N)·96` MACs/node.

| N (support) | hexion GFLOPs/eval | restnet (fixed) |
|---|---|---|
| 300 (opening) | ≈ 0.8 | 5.7 |
| 600 | ≈ 1.8 | 5.7 |
| 1000 | ≈ 3.4 | 5.7 |
| 1500 (crossover ≈ 1400) | ≈ 6.0 | 5.7 |
| 3000 (marathon tail) | ≈ 17 | 5.7 |

Mean over the self-play length distribution (N mostly 600–1500) ≈ parity-to-2× cheaper, with an honest
super-linear tail on marathons — and unlike the crop, priors cover the *full* legal set at every length.

### 10.3 Batching / padding plan

Inference: evaluator chunks (cap `max_batch_size`), rows grouped by node bucket (multiples of 256), per-group
padded forward, results scattered back to request order. **No cuDNN convs exist in this model** — all ops
are gather/GEMM/sdpa/LN, which are shape-robust; bucketing exists to bound *memory and recompiles*, not the
925 ms cuDNN autotune cliff (that hazard is specific to conv2d kernels, `inference.py:85-104`). Training:
micro-buckets per §7.3.

### 10.4 fp16 / compile / TRT story

- Transport f16 (features bounded [0,1], loss-free; same gate argument as the plane buffer).
- Inference: autocast fp16 (softmax/LN stay fp32 under autocast policy) as default; measured opt-in pure
  fp16 with LN kept in fp32 islands. Mask value −1e9 saturation is safe (token keys, §3.3).
- torch.compile: `dynamic=True` with `mark_dynamic` on (B, N_pad); the graph is matmul/sdpa/gather — the
  static-buffer freezing machinery restnet needed for its bias cache is unnecessary (bias is recomputed
  functionally from coords each forward). Fall back to eager without correctness loss (sdpa already fused).
- TRT: **not planned.** Dynamic-N gather/embedding graphs export fragilely; the measured 2.4–2.7× TRT win
  was on a fixed-shape dense conv model. Revisit only if profiling shows the evaluator GPU-bound after
  fp16+compile (the brief's measured wall-clock is evaluator-bound at 84%, but most of hexion's evals are
  3× cheaper than dense at the same position count).

### 10.5 VRAM

Training (batch 32, AMP): weights+grads+Adam(fp32) ≈ 24 MB; activations at typical mix (≈35k padded nodes,
~40 saved tensors, fp16) ≈ 250–400 MB; shared bias `B̂` ≤ 160 MB by the `B_g·N_pad² ≤ 2e7` rule; transient
dq/dr/idx ≤ ~230 MB at the worst bucket. **Peak < 1.2 GB.** Inference (no-grad, 256-leaf flush bucketed to
≤32-row groups at large N): bias transient ≤ ~270 MB, everything else ≤ 100 MB. **Peak < 0.5 GB.** Both fit
the shared 12 GB envelope with the live training+self-play coexistence margin intact.

---

## 11. Milestones (ordered; each gate blocks the next; no GPU-scheduling decisions)

| # | deliverable | acceptance gate |
|---|---|---|
| M0 | geometry/d6/features (Python): support set, halo, canonical order, raster featurizer | D6 suite + halo-shell invariant green on 1k fuzzed states; featurizer ≤ 2 ms/row at N=1500 |
| M1 | model: blocks, tokens, bias, heads; eager forward/backward | sdpa-vs-oracle ≤1e-5; padded-vs-single-row oracle; micro-bucket == monolithic grads; train==eval bit-parity; param count == §10.1 |
| M2 | Rust facts + featurizer + eval payload; Python evaluator | Rust/Python featurizer byte-equal; evaluator round-trip == direct forward; malformed-payload fail-loud tests |
| M3 | search: tree, lockstep + continuous, PCR/policy-init, noise, reuse, cache, TSS toggle | mix_seed/PCR/policy-init fixture parity vs dense_cnn streams; PUCT invariants (visit conservation, widening counts, determinism per seed); TSS on/off behavioral tests; CPU throughput probe recorded |
| M4 | data pipeline: shards, replay/shuffle, trainer hooks, plugin registration | overfit gate: 100 frozen games → policy CE < 0.05 nats above floor, value CE ≈ class entropy; shard round-trip; pipeline smoke run (tiny config) end-to-end through `hexo_train` |
| M5 | BC prefit (Phase A) + probe harness | §8 gates (top-1 within 2 pts of restnet prefit reference; ECE ≤ 0.08); probe npz persisting per epoch |
| M6 | RL smoke: self-play → shuffle → train → sealbot eval loop on a short config | epochs complete unattended; nan_trips == 0; evaluator GPU batch ≥ 32 effective; diagnostics contract (§7.5) fully populated |
| M7 | regression freeze + eval ladder definition | full test suite + fixtures frozen; anchored reference-checkpoint ladder configured (evaluation.py semantics) for later strength runs |

---

## 12. Envelope deviations (all from §3 refinable defaults; §2 honored without exception)

| id | deviation | justification |
|---|---|---|
| D1 | **LayerNorm (pre-norm) replaces BatchNorm in conv blocks**, and the block is pre-norm residual rather than post-activation | the variable-N case against BN: batch-statistics coupling across rows keyed on game length; train≠eval function under BN; running-stats lag at the BC→RL switch; micro-batch exactness (§7.3) and train==eval parity become theorems; no fold/fuse machinery. §3 itself names LN as the sanctioned fallback — promoted to default. The owner-locked item (§2.3) constrains the conv *operator* (7-tap direction-typed, preserved exactly), not the norm |
| D2 | one shared rel-pos table across the 3 A layers (per-layer tables = config knob) | per-row bias matrices (variable geometry) make per-layer gathers the dominant training memory; sharing → one (B,4,S,S) tensor per forward; 916 params; marginal expressivity loss at depth 3 |
| D3 | zero-init of each residual branch's last matrix (identity-at-init) | parameter-free stability: perfectly conditioned start, no depth-9 warmup fragility; standard practice; reversible by flag |
| D4 | per-node Linear stem instead of a conv stem | C1–C3 already give radius 6 before attention (the §3 requirement); fewer params; one less op family |
| D5 | linear LR warmup, first 500 optimizer steps of a fresh init only | Adam second-moment burn-in insurance at negligible cost; resumes skip it |
| D6 | policy/opp logits computed only at legal nodes (structural masking) | loss-equivalent to restnet's masked CE but identical-by-construction to the serve-time support; deletes the −1e9 fill and its fp16 caveats from the loss path |
| D7 | opp-policy target projected onto the current legal set + renormalized + row-masked when empty, with a coverage diagnostic | the head has no off-legal logits to absorb out-of-support mass (consequence of D6); measured, not assumed (≈tiny dropped mass; `opp_target_coverage_mean`) |
| D8 | phase stored as u8 enum in shards (vs object-dtype strings) | smaller, faster, pickle-free; schema is new anyway |

No §2 contradictions found. Two §2-adjacent notes, flagged for visibility: (a) hard-z is kept as the
default and only shipping value target; the soft-z knob exists but is procedure-gated (§4.2) per the
relocation-shock lesson; (b) "8 or 9 layers" → 9 chosen (the §3 interleave), with the C-heavy prefix
preserving the crop-conv local-reasoning character the owner trusts.
