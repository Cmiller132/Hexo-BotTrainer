# PLAN — hexfield_eq: register lane (sigmoid token writes) + ray-based line attention

Status: DRAFT / proposal. Date: 2026-07-08. Author: architecture design session
(follows the 4-agent review of the D6 rewrite). Target package:
**`packages/hexfield_eq`** (the ×12 D6-equivariant bot). The live `hexfield`
lineage is not touched.

**Prerequisite state (verified in-tree):** the D6 rewrite's Phase 3b is landed and
test-green — tied convs (`equivariant.py:gen_conv_weight`), group-norm with
orbit-tied affine, coset heads (`ATTENTION_HEADS=3` enforced,
`constants.py:179-183`), joint `(row, head)` bias tie
(`model.py:849-855,1029-1038`), full-net equivariance test passing for all 12
group elements. This plan builds directly on that trunk.

**Explicitly rejected direction:** KataGo-style global pooling
(the abandoned main_12 global-pooling plan) is NOT a consideration for this lineage
(owner decision, 2026-07-08). Every global mechanism below is attention-native.

---

## 0. Motivation (compressed — the design conversation's conclusions)

1. **Global bandwidth bottleneck.** Tokens are written only at attention blocks
   and frozen through every conv stretch (`model.py:1230-1240`). Cells receive
   global context only at A blocks. Value-relevant board-wide state is recomputed
   from scratch 3× per forward instead of accumulated.
2. **Softmax cannot count.** Softmax attention aggregates by weighted mean — it
   can *locate* the sharpest threat but cannot compute *how many* live threats
   each side has. Hexo (two placements per turn) is decided by threat
   *arithmetic*: you win by exceeding the opponent's per-turn answering capacity.
   The trunk currently has no sum-shaped aggregation primitive anywhere.
3. **Threats live on lines.** The decisive spatial structure is the length-6
   window along the 3 win axes. Full attention pays O(N²) to discover line
   structure the rules hand us for free; hex convs (radius 1) are too local to
   span it. The coset head structure (heads ≡ the 3 win axes, forced by the
   derivation) makes axis-restricted attention *exactly* equivariant — this is
   the one architecture family the D6 rewrite makes cheap to do correctly.

Two mechanisms, phased independently:

- **R — register lane:** every non-A block gets a cheap one-way, sigmoid-gated
  (sum-aggregating) token refresh. Tokens become a persistent, counting-capable
  global workspace updated ~11×/forward instead of 3×.
- **L — ray attention:** a third block type `L` whose attention is masked to
  game-live rays: from each cell, along each axis, out to window reach (5) and
  truncated at blocking stones. Own-threat and opponent-threat ray families get
  separate head triples.

---

## 1. Design decisions (locked unless marked open)

### Register lane

- **R1 — Aggregation is an unnormalized sigmoid-gated sum.**
  `upd = Σ_i m_i · σ(q·k_i/√d + b_t) · v_i · REG_SUM_SCALE`. No softmax
  normalizer: a token accumulates "number of cells matching pattern q" — the
  counting primitive softmax lacks. `REG_SUM_SCALE = 1/32` (fixed constant in
  `constants.py`, not env): typical matched-set sizes are tens of cells, so
  updates land O(1)–O(10). The sum runs in **fp32** under autocast (same
  rationale as the bias grad path).
- **R2 — Per-token gate threshold, head-constant.** `b_t: Parameter(NUM_TOKENS,)`
  added pre-sigmoid, broadcast over heads. Equivariance forces head-constancy:
  tokens carry no position, so their score rows must satisfy the `S_o = D6`
  case of the joint-tie analysis (bias constant across the head orbit —
  DERIVATION §5.2/§6). Init `-1.0` (gates ≈ 0.27; low-stakes since R3 gates the
  whole lane anyway).
- **R3 — Zero-init grow-in.** `out_proj` weight and bias zero-init ⇒ the lane is
  numerically a no-op at step 0. `REG_LANE=0` removes the modules entirely (true
  toggle-off byte-identity with the vanilla eq net).
- **R4 — cells←tokens read is a separate toggle (`REG_TOK_READ`), default OFF.**
  `x += Σ_t W_t·tokens_t` broadcast (per-token tied 1×1s, zero-init), applied at
  block **entry** (write happens at block **exit**, reading the block's output).
  Ships as its own A/B arm; cells already get global context at A blocks, so
  this must earn its place.
- **R5 — Same head machinery as main attention.** 3 coset heads,
  `head_dim = 4·C_ORBIT`, reusing `_eq.head_perm()`; `q/k/v/out` are
  `EquivLinear`. Pre-norm hygiene: dedicated `_make_norm` instances on the k/v
  input (cells) and q input (tokens); the summed update is added **raw** to the
  token residual stream — normalizing it would destroy the count magnitudes,
  which is the point of R1. Token magnitudes are re-normalized anyway at the
  next A block's `ln1` (pre-norm over the joint sequence).
- **R6 — Attached to every non-A block** (all C blocks now; L blocks too in
  Phase L). Tokens before the first A block are the learned slot-constant init
  (`model.py:1216-1221`) — a meaningful query source from block 0.
- **R7 — No scalar side-channel in v1.** An explicit √N size input was
  considered and dropped: sum aggregation already carries count/size
  information, and this keeps the lane free of anything pooling-shaped. Revisit
  only if prefit shows late-game value miscalibration that tracks support size.
- **R8 — No reduced-dim K/V in v1.** Full-width k/v projections cost
  ~2·N·C² ≈ 15% of a conv block. A `REG_KV_ORBITS` thinning knob is deferred to
  the Phase-L3 perf pass if serve throughput demands it.

### Ray attention

- **L1 — Ray definition (side-relative, from the side to move).** For cell `x`,
  axis `a ∈ {Q,R,QR}`, direction `± a.vector()`, side `s ∈ {own, opp}`:
  walk `j = 1..5`; at `y = x + j·dir`:
  - `y` off the support node set → stop (unattendable; geometric truncation);
  - `y` holds an **anti-s stone** → **include `y` (terminal blocker), then stop**;
  - else include `y`, continue. Stop after `j = 5` regardless.

  Reach 5 is not a tuning knob: a length-6 window through `x` extends at most 5
  cells along the axis, and a contiguous window containing both `x` and a cell
  beyond an anti-s stone necessarily contains that stone — so the ray set is
  *exactly* "cells sharing ≥1 clean-for-s window with x", plus the first blocker.
  Including the blocker is deliberate (deviation from the owner sketch, see §5):
  seeing the capping stone tells the head the line is dead-ended, which is
  tactical information the truncated-away cells can't carry.
- **L2 — Wire encoding is ray *lengths*, not index lists.** Per cell:
  `raylen: u8[12]` = (2 sides × 3 axes × 2 directions), values 0–5 (count of
  included cells, terminal blocker counted). The attention mask is reconstructed
  on-GPU from the pair `(dq, dr)` machinery the bias path already computes
  (`_build_pair` / the flex score_mod closures): pair `(i, j)` is live for
  `(s, a)` iff `(dq,dr)` is `a`-aligned with offset `k`, `1 ≤ |k| ≤ 5`, and
  `|k| ≤ raylen[i, s, a, sign(k)]`; the diagonal (self) is always live. This is
  12 bytes/cell on the wire vs ~120 for index lists, reuses existing machinery,
  and keeps the Rust side a trivial walk.
- **L3 — L block = cells-only ray attention + the register refresh.** Tokens do
  not join the L sequence (they interact via the register lane and A blocks);
  seq length stays `Npad`, shapes uniform with C blocks. Residual/MLP structure
  mirrors `AttnBlock` (pre-norm, LayerScale) minus the token rows.
- **L4 — 6 heads = 3 cosets × 2 orbit-halves (own/opp).** Each win-axis coset's
  channel block is `(|K|=4 slots) × C_ORBIT`; the group action permutes the K
  slots and is **identity on the orbit index**. Therefore the own/opp sub-head
  split must be along the **orbit index** (`C_ORBIT → C_ORBIT/2 + C_ORBIT/2`),
  NEVER along the K slots — K acts transitively on itself, so any slot split
  breaks equivariance. This is the plan's sharpest correctness trap; it gets its
  own unit test. Constraints: `C_ORBIT` even (16 ✓); `head_dim_L = 2·C_ORBIT`
  (32 at c=192 — on the kernel fast path). Needs a `head_perm6` variant in
  `equivariant.py` (coset-major, then orbit-half) and a relaxation of the
  `ATTENTION_HEADS == 3` import check (`constants.py:179-183`) to "A blocks 3;
  L blocks 6" (per-block-type head counts).
  - *Fallback if this proves fiddly:* 3 heads with the union mask (own ∪ opp
    rays), losing side-specificity — acceptable but strictly weaker; the
    own/opp separation is what lets a head specialize in "my threats" vs "their
    threats".
- **L5 — Bias: reuse the joint tie, extended by the side index.** All ray
  offsets lie in the exact disk (dist ≤ 5 < 8), so L blocks reuse the existing
  orbit/joint LUTs; the side (own/opp) index is group-invariant, so
  `bias_theta_L: Parameter(n_joint_classes, 2)` indexed
  `[joint_of_row_head, side]` is equivariant with no new derivation needed.
  Masked (non-ray) pairs get the additive `PAD_KEY_MASK_VALUE = -3e4`
  convention, not `-inf` (fp16 safety, matches `model.py:53` convention).
- **L6 — Geometric-ray ablation toggle.** `RAY_BLOCKERS=0` disables blocker
  truncation (pure axis-disk-5 axial attention, computable from coords alone).
  This is the attribution control separating "line-restricted attention helps"
  from "game-semantic truncation helps".
- **L7 — Sequencing: L phases start only after the register-lane arms have a
  prefit verdict**, and both ride *after* the eq plan's own Phase-4 vanilla
  BC-prefit gate (the vanilla net must pass its go/no-go first, unmodified, so
  a miss stays attributable to features/equivariance/width per the eq plan's
  diagnosis axes).

### Trunk layout

- **T1 — Phase R baseline: keep `CCCACCCACCA`** (the current default,
  `constants.py:204` — 8C + 3A, 11 blocks, ends in A). This is the owner's
  "CCCA-first" instinct and it is already the package default; with the
  register lane, tokens update at all 11 blocks, which is precisely what makes
  the long conv runs affordable.
- **T2 — Phase L primary: `CCLACCLACLA`** (5C + 3L + 3A, depth 11 unchanged).
  Rationale for the position within each run: two convs build local features to
  radius 2 (the near-window neighbourhood, already primed by the graded window
  input planes), L then routes content along live rays (reach 5), A globalizes.
  Cautious arm: `CCLACCLACCA` (2 L). Control: `CCCACCCACCA` stays one env var
  away. Layout grammar gains `L` (`constants.py:205-213` validation; must still
  end in `A`).
- **T3 — Depth/width changes are out of scope.** Same depth (11) and width
  (D5's choice at the eq prefit gate) across all arms, so every A/B isolates
  one mechanism.

---

## 2. Architecture spec

### 2.1 `RegisterRefresh` module (Phase R)

```python
class RegisterRefresh(nn.Module):
    """One-way sigmoid-gated cross-attention: tokens read cells, SUM-aggregated
    (counting-capable). Zero-init out_proj => no-op at step 0."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = ATTENTION_HEADS                    # 3 coset heads
        self.head_dim = channels // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.ln_kv = _make_norm(channels)               # pre-norm on cells
        self.ln_q = _make_norm(channels)                # pre-norm on tokens
        self.q_proj = linear(channels, channels)
        self.k_proj = linear(channels, channels)
        self.v_proj = linear(channels, channels)
        self.out_proj = linear(channels, channels)      # ZERO-INIT (weight+bias)
        self.gate_bias = nn.Parameter(torch.full((NUM_TOKENS,), -1.0))  # head-constant (R2)
        if EQUIVARIANT:
            self.register_buffer("_head_perm", _eq.head_perm(), persistent=False)
            self.register_buffer("_head_perm_inv", _eq.head_perm_inv(), persistent=False)

    def forward(self, tokens, x, mask):                 # (B,T,C), (B,N,C), (B,N)
        q = self.q_proj(self.ln_q(tokens))
        k = self.k_proj(self.ln_kv(x))
        v = self.v_proj(self.ln_kv(x))
        # coset-permute channels, reshape to (B, h, T/N, d) exactly as
        # RelPosAttention does (model.py:642-654)
        ...
        scores = (q @ k.mT) * self.scale + self.gate_bias.view(1, 1, -1, 1)
        gates = torch.sigmoid(scores.float()) * mask[:, None, None, :]   # fp32 (R1)
        upd = (gates @ v.float()) * REG_SUM_SCALE       # SUM over cells — counts
        # merge heads, un-permute cosets, project; raw residual add (no norm — R5)
        return tokens + self.out_proj(merge(upd).to(tokens.dtype))
```

Integration in the trunk walk (`model.py:1230-1240`):

```python
if kind == "C":
    if REG_TOK_READ:
        x = x + self.tok_read[ci](tokens) * mask.unsqueeze(-1)   # entry, zero-init
    x = self.conv_blocks[ci](x, gather_idx, mask)
    if REG_LANE:
        tokens = self.registers[ci](tokens, x, mask)             # exit
    ci += 1
```

`tokens` thus becomes a loop-carried variable through conv stretches (today it
is only reassigned at A-block splits). The A blocks are untouched.

Cost at c=192, N≈400: k/v/out GEMMs ≈ 3·N·C² ≈ 44M MACs (~15–20% of a conv
block's 206M); score/aggregation `T·N·C` ≈ 0.5M — negligible. Serve: plain
GEMMs + sigmoid, fp16-safe (aggregation upcast), CUDA-graph and
`torch.compile(dynamic=True)` friendly (no data-dependent shapes).

### 2.2 `RayAttnBlock` (Phase L)

```python
class RayAttnBlock(nn.Module):
    """Cells-only attention masked to live rays. 6 heads = 3 win-axis cosets
    x {own, opp} orbit-halves (L4). Pre-norm residual + MLP like AttnBlock."""
    # q/k/v/out: EquivLinear; channel order via head_perm6 (coset, orbit-half)
    # bias: bias_theta_L (n_joint_classes, 2)[joint_of_row_head, side]
    # mask: additive -3e4 where pair (i,j) is not on a live ray for (side, axis)
```

Mask construction rides the existing pair machinery: the materialized reference
path extends `_build_pair` (which already computes per-pair `(dq, dr)`) with the
axis-alignment + `raylen` comparison of L2; the train/serve flex paths get a
`score_mod` closure over `(coords, raylen)` exactly parallel to the existing
`_FlexBias` carriers. The bespoke Triton pair kernel is NOT extended in v1 — L
blocks run the flex path on serve until Phase L3 measures whether a gathered
local-attention kernel (≤ 61 keys/query) is worth writing.

### 2.3 Rust / featurizer / wire (Phase L0)

- `features.rs`: `ray_lengths(board, support) -> u8[N][12]` — the L1 walk;
  shares the board access `window_feature_row` already uses. Serve packs it as a
  new ABI buffer (`payload.rs`, `serve_pack.rs`, `batching.py` — same pattern as
  `nbr`).
- `replay_expand.rs`: identical walk on the reconstructed board (the
  `WindowStore`/board rebuild already exists at `replay_expand.rs:441-468`).
  Derived data — **no shard schema change** (recomputed from placements, like
  the graded planes).
- `features.py`: Python oracle `ray_lengths_for_cell` for the 3-way parity
  tests.
- Equivariance of the data: rays are recomputed from the (transformed) board, so
  the all-12 D6 parity harness (`tests/test_hexfield_eq_rust_parity.py`
  pattern) extends directly: `raylen[g·x, s, σ_g(a), dir_g] == raylen[x, s, a, dir]`
  with the reflection direction-swap handled by the axis-permutation LUT.

---

## 3. Phased plan (each phase gated; arms are cumulative toggles)

### Phase R0 — register lane, code + correctness
- `constants.py`: `HEXFIELD_REG_LANE` (default 0 until R2 verdict),
  `HEXFIELD_REG_TOK_READ` (default 0), `REG_SUM_SCALE=1/32` constant.
- `RegisterRefresh` + `tok_read` modules; trunk-walk threading; checkpoint
  `meta` records `reg_lane/reg_tok_read` (rides the meta-first arch work — a
  hard prerequisite here, since `KNOWN_TRUNK_LAYOUTS`' (conv, attn) counts
  cannot express register/L variants).
- **Param-grouping predicates** (`plugin.py`, `trainer.py`, `prefit.py`): new
  names classified — `registers.*.{q,k,v,out}_proj` decay; `ln_*` affine +
  `gate_bias` no-decay; grad-norm group for the lane. This is the known
  silent-corruption trap — update all three in lockstep.
- **Gate:** full-net equivariance test green with `REG_LANE=1` (all 12 g,
  randomized params); toggle-off byte-identity vs vanilla; zero-init
  forward-identity at step 0; grads reach every lane param; a synthetic
  counting probe (duplicate a matched pattern k× on a board → token update
  scales ≈ k) passes.

### Phase R1 — prefit A/B (the lane's verdict)
- Arms from the same split/seed: (1) vanilla eq [exists, the eq plan's Phase-4
  gate run], (2) +`REG_LANE`, (3) +`REG_TOK_READ`. Zero-init ⇒ each arm starts
  numerically at its predecessor.
- **Gate:** arm 2 ≥ arm 1 on held-out top-1 and `value_ece`; look for the value
  win specifically (calibration by ply bucket, search-vs-net value gap on the
  probe set). **Kill criterion:** arm 2 ≤ arm 1 on both → `REG_LANE=0` and stop
  (the lane is one env var; nothing else in this plan depends on it). Arm 3
  ships only if it beats arm 2.

### Phase L0 — ray data (Rust + wire + parity)
- `ray_lengths` in `features.rs` + `replay_expand.rs` + Python oracle; ABI
  buffer; rebuild via maturin.
- **Gate:** 3-way parity (serve Rust / train Rust / oracle) exact; all-12 D6
  raylen-permutation test green; wire round-trip test.

### Phase L1 — L block, code + correctness
- `head_perm6` + inverse in `equivariant.py` (with the L4 orbit-half unit test:
  assert the permutation conjugates the regular action to a (coset ⋊ K-slot)
  block structure preserving the halves); relax the head-count import check to
  per-block-type; `RayAttnBlock` (materialized reference + flex score_mod);
  `bias_theta_L`; layout grammar `L`.
- **Gate:** full-net equivariance with an L layout (all 12 g — this is the test
  that catches a wrong `head_perm6` or a slot-split mistake);
  materialized ≡ flex parity; masked-softmax never sees an empty row
  (self-inclusion test on a lone stone board); grads reach `bias_theta_L`.

### Phase L2 — layout prefit A/B
- Arm (4): `CCLACCLACLA` (+ best-of R arms); arm (4b cautious): `CCLACCLACCA`;
  arm (4c control): `RAY_BLOCKERS=0` geometric rays.
- **Gate:** arm 4 ≥ best R arm on prefit metrics; 4 vs 4c attributes the
  blocker semantics. **Kill:** L regresses → stay on `CCCACCCACCA` (layout is
  env-only; the ray data path stays for future use).

### Phase L3 — serve perf pass
- Measure: L-block flex path throughput vs C block; register-lane serve
  overhead; end-to-end live serve rate vs budget.
- Optimize only what measures hot: gathered local-attention Triton kernel for L
  (≤ 61 keys/query), `REG_KV_ORBITS` thinning (R8). Parity gate (3e-3) for any
  kernel added.

### Phase S — soak
- Winning arm into a short self-play soak; entropy/length/calibration bands;
  then the standard multistage eval vs Strix/SealBot/self-anchors.

---

## 4. Cross-cutting

- **Checkpoint meta is load-bearing** (shared requirement with the eq plan's
  outstanding item): `trunk_layout` string, `reg_lane`, `reg_tok_read`,
  `ray_blockers`, L-head config must be persisted in `meta`/`extra` and read
  meta-first by BOTH arch inferers (`model.infer_net_kwargs_from_state_dict`
  and the dashboard's `debug_infer.py`). `KNOWN_TRUNK_LAYOUTS` cannot represent
  these variants — do not extend it; meta-first supersedes it for this lineage.
- **Param-name predicates** must track every new module (R0/L1 both) —
  `plugin.py` no-decay, `trainer.py` grad-norm groups, `prefit.py`.
- **Env prefix:** new knobs follow whatever resolution the package adopts for
  the `HEXFIELD_*` collision issue (open item from the review); until then they
  use `HEXFIELD_REG_*` / `HEXFIELD_RAY_*` for consistency with the existing
  convention.
- **Tests to add:** equivariance (lane on; L layout), toggle-off byte-identity,
  zero-init identity, counting probe, `head_perm6` structure test, raylen 3-way
  parity + all-12 D6, empty-ray softmax safety, materialized≡flex L parity.

## 5. Deviations from the owner's sketch (judgment calls, flag for sign-off)

1. **Terminal blocker included in the ray** (owner: "until they hit an opponent
   stone" — ambiguous on inclusion). Included: the capping stone is tactical
   information ("this line is dead"), and excluding it costs a query the only
   direct view of *why* its line is capped.
2. **Ray lengths on the wire, not index lists** (12 B/cell vs ~120; mask
   rebuilt on-GPU from the existing pair `(dq,dr)` machinery).
3. **Own/opp side split as separate head triples** — the sketch had one ray
   family; side-relative rays differ (own rays pass through own stones, stop at
   opp stones; vice versa), and a single family can't represent both
   perspectives without content re-derivation. The 6-head orbit-half split (L4)
   is the equivariant way to host both.
4. **Reach hard-capped at 5** — not a knob; it is the window-6 geometry
   ("no longer able to complete a winning line" made exact).
5. **L blocks are cells-only** — tokens interact via the register lane and A
   blocks; keeps L shapes uniform and the mask assembly simple.
6. **No scalar size input to the lane (v1)** — sum aggregation already carries
   count/size; keeping the lane free of pooling-shaped inputs per the owner's
   direction.

## 6. Risk register (ranked)

1. **`head_perm6` / orbit-half split correctness (L4)** — a K-slot split is a
   *silent* equivariance break. Guard: the dedicated structure unit test + the
   randomized-param full-net equivariance test with an L layout (the test class
   that caught nothing being wrong in 3b is the same one that would catch this).
2. **Sum-aggregation scale drift** — unnormalized sums can grow with support
   size; fp16 overflow or LR sensitivity. Mitigations: fp32 aggregation,
   `REG_SUM_SCALE`, zero-init grow-in, A-block pre-norms re-normalizing the
   token stream. Watch the lane's grad-norm group during prefit.
3. **Flex score_mod capacity/perf for the ray mask** — the closure adds a
   raylen gather + axis tests per (q,k) pair. If flex compilation or throughput
   disappoints, the materialized masked-sdpa path is the fallback at O(N²)
   memory; the gathered kernel (L3) is the fix, not a prerequisite.
4. **Attribution creep** — five toggles compound. Discipline: the cumulative
   arm ladder (§3), one mechanism per arm, zero-init so each arm starts at its
   predecessor.
5. **Wire ABI bump (raylen)** — serve/train desync class. Same pattern as the
   25-plane bump that just shipped cleanly; parity tests are the guard.
6. **Compute budget** — lane ≈ +15–20% per C block; L block ≈ A-block cost
   minus the N² savings from 61-key locality (flex path realizes only part of
   this until L3). Budgeted at the Phase-L3 measurement, with depth/width held
   fixed (T3).

## 7. References

- `docs/PLAN_D6_EQUIVARIANT_REWRITE.md` — the base rewrite (Phases 0–3b landed).
- `docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` — §4 (coset heads), §5 (joint
  bias tie; the `S_o = D6` head-constancy used by R2/L5), §6 (token rows).
- The abandoned main_12 global-pooling plan — the rejected pooling alternative
  (legacy lineage only; kept for the record).
- Code anchors: trunk walk `model.py:1160-1243`; ConvBlock `model.py:548-598`;
  RelPosAttention `model.py:601-694`; coset machinery + LUTs `equivariant.py`;
  layout/env `constants.py:96-213`.
