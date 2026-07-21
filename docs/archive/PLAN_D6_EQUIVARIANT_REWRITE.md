# PLAN — hexfield next-bot: D6-equivariant trunk + graded window features (full rewrite)

Status: DRAFT / proposal. Date: 2026-07-08. Author: architecture spike (4-agent
codebase survey). This is a **full rewrite / new bot**, not a refactor: old parts
are stripped, checkpoint back-compat is abandoned.

**Decisions locked (owner, 2026-07-08):** go **straight to full Tier 2** (one
rewrite — no intermediate non-equivariant milestone); **full D6, group order 12**
(regular rep ×12, augmentation not ported); new package **`packages/hexfield_eq`**
(own cdylib; the live `hexfield` lineage stays untouched); **re-anchor eval on
Strix + SealBot + new-bot self-anchors** (drop the 15-plane `main5`/`main6`
anchors). Because we skip the Tier-1 milestone, the equivariant-attention
derivation (Phase 3a) and the BC-prefit gate (Phase 4) are the two critical
go/no-go points with no fallback bot behind them.

> This plan folds three coupled workstreams from the next-bot design doc into one
> phased build:
> 1. **§3 Graded per-axis window features** — retire the 4 binary hot/win planes,
>    add 14 graded per-(cell,axis) planes; 15 → 25 input planes; computed in Rust.
> 2. **§4.1 Orbit-tie the bias table** — tie the 217 exact-disk rows over D6 orbits
>    (237 → 45 free rows). Shipped regardless of trunk tier.
> 3. **§2.3 Tier 2 D6-equivariant trunk** — regular-representation weight tying on
>    convs + attention, group-norm, group-pooled heads, and **delete D6
>    augmentation**. This is the high-risk / high-reward change; §2.4 makes clear
>    §3 and §4.1 are prerequisites/interacting parts.

All line refs are the current working tree (`main_9-fastrow-strip`). The design
doc's own `trainer.py:469-522` ref is **stale** — the real D6 draw is
`trainer.py:615-617` (verified).

---

## 0. Feasibility verdict (what the survey found)

**Overall: feasible, and Tier 2 is more feasible than the doc implies on the
kernel side, but harder than the doc implies on two model-side points.** Summary
by component:

| Component | Verdict | Why |
|---|---|---|
| Orbit-tied bias table | ✅ **Clean** | Kernels only gather `table[row,h]`; tying happens in torch upstream, 237-row shape preserved. Verified **25 disk orbits → 45 free rows**. Zero kernel change. (`model.py:617-622`, `_triton_attn.py:166-171`) |
| Equivariant Q/K/V/out projections | ✅ **Clean reparam** | They are plain `nn.Linear` in torch (`model.py:451-454`), never enter Triton. Tying is a pure weight reparameterization. |
| Tied conv weights (regular rep) | ✅ **"Forward unchanged" holds** for the reference GEMM (`model.py:370-374`) and the fp16 Triton conv/conv+LN kernels — they consume a **materialized** `(7,C_in,C_out)` weight and are blind to orbit structure. Generate the full weight **eagerly** inside `HexNodeConv.forward`. | 3 gotchas below. |
| Graded window features (Rust) | ✅ **API exists** | `WindowStore::entry`, `WindowEntry::mask/count/empty_mask/key`, `Axis::ALL/vector`, `Board::windows()` all public (`hexo_engine/rust/src/tactics.rs`, `board.rs:120`). One plumbing gap (train-time kernel has no store) — small engine method fixes it. |
| Delete D6 augmentation | ✅ **Low-risk, mostly Python** | `d6 = zeros` makes every backend an identity no-op (one-line disable); full removal is mechanical Python+Rust cleanup. |
| **Equivariant attention (full block)** | ⚠️ **Needs a design derivation first** | Orbit-tied bias gives bias *invariance* but the interaction of regular-rep fibers × head split × per-head bias is under-specified. **Top risk** — needs a group-theory spike before Phase 4. |
| **LayerNorm / LayerScale equivariance** | ⚠️ **Pervasive** | 8+ LN sites + every `LayerScale.gamma` apply a dense per-channel affine that *breaks* equivariance; each must become group-norm with **orbit-tied affine**. |
| Cross-arch eval vs old anchors | ⛔ **Blocked by process-global featurizer** | A 25-plane bot cannot serve a 15-plane `main5`/`main6` anchor in one process (stem shape mismatch). Must re-anchor. |

### The three tied-conv gotchas (all manageable)
1. **fp8 conv weight cache** (`_triton_conv.py:359-374`) is keyed by `id(weight)`
   → per-forward regeneration causes miss-every-forward + an unbounded strong-ref
   leak. Only bites under `HEXFIELD_CONV_FP8=1`. **Fix: drop fp8 for v1** (revisit
   later) or re-key the cache on base-param `_version`.
2. **`SERVE_HALF` deep-copy-and-half** (`inference.py:381-397`) freezes weights
   once at serve init → weight generation must live **inside** the module forward
   (it already does). Add a **dense-weight cache keyed on base-param `_version`**
   so frozen-serve weights regenerate once, not per forward.
3. **Channel alignment**: `C = g·C_orbit` needs `C % 16 == 0` (so `C_orbit`
   divisible by 4 at g=12 / by 8 at... — pick widths deliberately) and a head
   count landing `head_dim ∈ {16,32,64,128}`, else convs silently fall off the
   Triton fast path (`model.py:356-361`).

---

## 1. Architecture target (the new bot)

### 1.1 Input: 25 planes (11 kept scalar + 12 axis-indexed + 2 scalar fork)

Retire planes 9/10/13/14 (`F_OPP_HOT/F_OWN_HOT/F_OPP_WIN_NOW/F_OWN_WIN_NOW`,
`constants.py:50-55`) and the `HOT_MIN_PLACEMENTS` gate. Add, per support cell `x`
and axis `a ∈ {Q,R,QR}` over the 6 length-6 windows through `x` on `a`:

- `own_line[a]`, `opp_line[a]` — max own/opp count among clean windows (empty at
  `x` for empties), /5. (**4 planes** counting own+opp × ... no: 2 planes × 3 axes.)
- `own_live[a]`, `opp_live[a]` — count of clean-for-side windows (openness), /6.
- → **12 axis-indexed planes** (4 quantities × 3 axes).
- `own_fork`, `opp_fork` = |{a : line[a] ≥ 3}| / 3 — **2 scalar planes**.

Total new = 14; kept = 11; **NUM_FEATURES = 25**. For stones (non-empty cells)
compute with "empty at x" dropped. Everything is derivable from the stored
placement history — **no new shard facts needed** (the four hot/win CSR columns
become obsolete and are recomputed at train time).

**Off-board window openness — an OWNED design decision, not a bug.** A length-6
window through an edge cell that runs off the legal support is counted as
clean-and-empty: an absent cell contributes 0 own and 0 opp stones (the `None ⇒
counts 0 ⇒ clean+empty` rule of Phase 1.2's `window_features`). So an edge cell's
off-board windows still contribute to `own_live`/`opp_live`, even though such a
window can never actually complete a line — edge cells therefore read slightly
"more open". This is **deliberate and player-symmetric** (own and opp are treated
identically), and both featurizers (the Rust `WindowStore` path and the Python
oracle) implement it the same way, so they stay in exact parity. The featurizer
does *not* distinguish an off-board cell from an interior empty cell. The
alternative — masking off-board windows out of the openness count — is a possible
future feature-quality tweak, **not** a correctness fix; the project owns the
current behavior explicitly (BUGS_FOUND.md closes this as design, not defect).

### 1.2 D6 typing of the input (needed for Tier 2)

Under D6 the three axes `{Q,R,QR}` form the **S3 permutation module** (rot60
3-cycles Q→R→QR; reflection transposes two axes — `geometry.py:89`,
`DIRECTIONS` `constants.py:20-27`). So the 25 planes carry types:
- 11 kept planes + 2 fork planes = **13 scalars (trivial rep)** → copy into all
  fiber slots.
- 12 axis-planes = **4 copies of the 3-dim axis-permutation module** → lift by the
  axis 3-cycle/transposition action.

### 1.3 Trunk

- Channels `C = g · C_orbit` where `g` is the group order (**12 for full D6**, or
  6 for C6-rotations-only with reflection left to augmentation — see Decision D1).
- **HexNodeConv (tied):** store `w_base:(7, C_in_orbit, g, C_out_orbit)`;
  generate the full `(7,C_in,C_out)` weight each forward by a precomputed
  index-gather (group action permutes the 6 direction taps + cyclically relabels
  the g-axis; reflection reverses the tap cycle). Reference GEMM + Triton kernels
  consume the generated weight unchanged.
- **Group-norm** replaces every `nn.LayerNorm(C)`: normalize over the full fiber
  structure but with **orbit-tied affine** (one scale/shift per `C_orbit`,
  broadcast over the g slots). Same for `LayerScale.gamma`.
- **Equivariant attention:** Q/K/V/out and the MLP `fc1/fc2` become tied
  block-structured 1×1s on the fiber; the rel-pos bias is orbit-tied; summary
  tokens carry the regular rep (learned `(NUM_TOKENS, C_orbit)` broadcast over
  fibers). **The head-axis × fiber × bias interaction is a design spike (Phase 4a).**
- **Orbit-tied bias table** (ships regardless of tier): per block, a `(45, heads)`
  free table + a `register_buffer` `orbit_of_row:(237,)` LUT; generate
  `bias_table = free_rows[orbit_of_row]` at the top of each bias build. Downstream
  (`_build_pair`, `build_attn_bias`, flex carriers, `_BiasGather`, Triton attn) is
  unchanged. Keep `_BiasGather` on the expanded 237-row table and let the
  `free_rows[orbit_of_row]` index-select carry gradients back to the 45 params.

### 1.4 Heads

- policy / opp_policy / soft_policy / cell_q are **per-cell covariant** — but each
  final `Linear(C,·)` must be a **fiber-invariant read** (group-pool the fiber →
  `C_orbit`, then reduce) so the per-cell scalar/logit lands in the trivial rep.
- value / stvalue_* / moves_left need **invariant** inputs: **group-pool** (mean
  over the g fiber slots) each of the token and pooled-cell vectors *before* the
  `3C→C` reductions (`_value_input` `model.py:1012-1016`, `_pooled` `938-942`
  gains a fiber-mean).
- Prune dead capacity: tokens 6 & 7 are read by no head → `NUM_TOKENS` 8 → 6.

---

## 2. Decisions (resolved) + residual open points

**Resolved (owner, 2026-07-08):**
- **D1 — Group order = full D6, ×12.** The regular rep is order 12 (6 rotations ×
  reflection). Augmentation is not ported; reflection is a first-class part of
  every tied op, group-norm, and head-pool (tap-cycle reversal + axis transpose +
  fiber relabel).
- **D2 — Straight to full Tier 2.** One rewrite; no intermediate non-equivariant
  milestone. Consequence: the A/B "matched control" is produced on demand as an
  **in-package ablation** (`HEXFIELD_GROUP_ORDER=1`), not a separately shipped bot
  (see D6 residual).
- **D3 — New package `packages/hexfield_eq`** (own cdylib, own maturin crate). The
  live `hexfield` lineage, its eval anchors, and the dashboard loaders are left
  intact. The one shared change (the engine `WindowStore` constructor, Phase 1.1)
  lands in `hexo_engine`; both packages link it.
- **D4 — Re-anchor on Strix + SealBot + new-bot self-anchors.** Drop the 15-plane
  `main5`/`main6` permanent anchors (`permanent_anchors = ()`).

**Residual open points (decide at the marked phase, not now):**
- **D5 — Target width (decide at Phase 4, the BC-prefit gate).** ×12 tying saves
  ~12× conv params; "spend it on width." Constraints: `C = 12·C_orbit`,
  `C % 16 == 0` (⇒ `C_orbit` divisible by 4), `head_dim = C/heads ∈ {16,32,64,128}`.
  Candidate: `C_orbit=16 → C=192, heads=3, d=64` (the existing fast-path sweet
  spot). Fix by matching the current c=128 param budget at the gate.
- **D6 — A/B control fairness (decide at Phase 6).** With no Tier-1 bot, the
  cleanest equivariance A/B control is a `GROUP_ORDER=1` ablation — but to be a
  *fair* control it needs augmentation (else it is non-equivariant AND
  non-augmented, strictly worse). Choice at Phase 6: (a) port a minimal
  augmentation path guarded to the ablation build only, or (b) skip the matched
  A/B and gate purely on **absolute strength vs Strix/SealBot** (accepting we
  cannot cleanly attribute the delta to equivariance vs the graded features).

---

## 3. Phased plan

Each phase has an acceptance gate; a later phase does not start until its
predecessor's gate is green. The build is **one target** (full ×12-equivariant,
25-plane, augmentation-free): Phase 0 scaffolds the new package; Phases 1–2 land
the feature + bias changes (validated in isolation); Phase 3 is the equivariant
trunk (spike then implement); Phase 4 is the BC-prefit **go/no-go**; Phases 5–6 are
eval/serve and the soak/ablation A/B.

### Phase 0 — Scaffolding & foundations
- Create `packages/hexfield_eq` skeleton (own `pyproject`, own cdylib, own maturin
  crate, entry point `[hexo_train.models]`), copying hexfield's module layout. The
  live `hexfield` package is not touched.
- Add arch env knobs read once at import in `constants.py` (matching the
  `CHANNELS`/`ATTENTION_HEADS`/`TRUNK` convention, `constants.py:81-99`):
  `HEXFIELD_GROUP_ORDER` (**default 12**; `1` = non-equivariant ablation for the
  Phase-6 A/B, `6` reserved), `HEXFIELD_C_ORBIT`, `NUM_FEATURES` a plain constant
  (=25). Enforce `C = 12·C_orbit`, `C % 16 == 0`, `head_dim ∈ {16,32,64,128}` at
  import.
- **Do not port the augmentation machinery.** The expand path is identity-only
  from day one (`symmetry=0` everywhere); the model's equivariance (Phase 3) is
  what makes this sound, and the Phase-4 equivariance test is its guarantee. (This
  replaces the old "build augmentation, then delete it" plan.)
- **Gate:** empty package builds and imports; `hexo_train` discovers the plugin;
  an e2e CPU smoke stub runs the epoch loop with a trivial net.

### Phase 1 — Graded window features (Rust-first, 15 → 25)
The largest cross-language change; independent of the trunk, validated in isolation
against a non-equivariant stub net before Phase 3.

1. **Engine method (shared, `hexo_engine`).** Add a public
   `WindowStore::from_placements(&[(HexCoord, Player)])` (or `Board::from_stones`)
   so the train-time expand kernel can build a store without full-state legality
   (the population methods are `pub(crate)`/`#[cfg(test)]` today — `tactics.rs:416,422`).
   Alternatively reconstruct a `HexoState` via public `new()` + `apply_placement`
   (`state.rs:151,399`) per row — **recommend the small public store constructor**
   (cheaper, matches the `window_features(board,…)` signature). Rebuild via maturin.
2. **`window_features` in Rust.** New fn (serve: `features.rs`, replacing
   `fill_hot_and_win` `79-120` and its call site `71`; train: `replay_expand.rs`,
   replacing the stored-cell fills `441-457`). For each support cell, each
   `Axis::ALL`, each `offset in 0..6`: `key.start = x − axis.vector()*offset`,
   `windows.entry(key)`; own count = `entry.count(me)`, clean = `entry.count(other)==0`,
   empty-at-x = `(entry.empty_mask()>>offset)&1`; `None` ⇒ counts 0 ⇒ clean+empty.
   Emit the 12 axis planes + 2 fork scalars with /5, /6, /3 normalizers.
3. **`constants.py:56` + `constants.rs:9`**: `NUM_FEATURES = 25`; add 14 `F_*`,
   delete 4 retired `F_*`; delete `HOT_MIN_COUNT/WIN_NOW_COUNT/HOT_MIN_PLACEMENTS`;
   keep `WINDOW_LEN`; add the normalizer constants.
4. **Python featurizer** (`features.py:build_features 165-210`): populate the 14
   new planes; a **new graded oracle** enumerating all 6 windows through each
   support cell (including all-empty windows — `window_scan 99-159` only visits
   stone-anchored windows and can't be reused as-is; keep it only as a test oracle
   or retire it).
5. **Shard schema bump** (`shards.py` SCHEMA_VERSION `38`): drop the 4 hot/win CSR
   groups from writer (`104-106,149-158,234-236`), reader (`382-385`),
   `window.py` (`81-84,95-98,347-350,310-313,148-151,172-182`), `samples.py`
   (`33-36,57-59`), and `replay_expand.rs` (`110-113,932-935,960-963,1036-1039`).
   `selfplay.py` producer (`41,312-325`) stops calling `window_scan`.
6. **Wire/serve buffers** follow the constant automatically (`inference.py`,
   `payload.rs`, `serve_pack.rs`, `batching.py:57` derives width from the array).
   Fix the one literal: `tests/test_hexfield_rust_parity.py:34`.
7. **Parity + D6 tests.** Regenerate `test_hexfield_rust_parity.py` (graded planes
   ride the `1e-6` tolerant-plane path, not exact-equality). Extend
   `tests/katago_buffer/test_p7_rust_parity.py::test_rust_equals_serial_all_d6`
   (`155-181`) with an explicit **axis-permutation assertion**: a rot60/reflect
   maps `own_line[Q]@x → own_line[σ(Q)]@σ(x)`.
- **Gate:** Python↔Rust featurizer parity exact (non-recency) / ≤1e-6 (graded);
  all-12-D6 expand parity green incl. axis permutation; shards round-trip at the
  new schema; a training epoch consumes 25-plane batches (stem
  `HexNodeConv(NUM_FEATURES,c)` `model.py:606` auto-widens).

### Phase 2 — Orbit-tied bias table
Independent of tier; ships in Tier 1.
- Precompute `orbit_of_row:(237,)` (25 disk orbits + 8+8 ring + 1 far + 3 token =
  45 classes) as a `register_buffer(persistent=True)`. Replace each block's
  `Parameter(237,heads)` (`model.py:617-622`) with `Parameter(45,heads)`; build
  `bias_table = free_rows[orbit_of_row]` at the top of `build_attn_bias`,
  `_build_pair_u8`, and the flex carriers.
- Keep `_BiasGather` (`300-322`) on the expanded 237-row table; gradients flow to
  45 params via the index-select. Verify the double-gather composes and the
  histogram backward still targets the right class count.
- **Param-grouping fix (silent-corruption class):** the AdamW no-decay predicate
  keys on the substring `"bias_table"` (`plugin.py:38`) and the grad-norm groups
  on `"bias_tables"`/`"tokens"` (`trainer.py:261-266`). Keep the new param name
  containing `bias_table` **or** update both predicates in lockstep.
- **Gate:** unit test — the expanded table equals a per-row-free table restricted
  to be orbit-constant; an equivariance micro-test — with a from-scratch tied bias,
  the attention scores are exactly D6-invariant on a probe board; training a few
  steps updates all 45 params.

### Phase 3 — Equivariant trunk (×12 regular representation)
The core of the rewrite. Built directly on the Phase 1–2 package (25 planes,
orbit-tied bias). No non-equivariant intermediate.

**Phase 3a — group-theory design spike (do first, NO code). Critical path — there
is no fallback bot behind it.** Derive how a board symmetry `g ∈ D6` acts jointly
on (regular-rep Q/K/V fibers, the `heads×head_dim` split, the per-head bias table)
so attention scores are equivariant. Deliverables, as a short derivation doc with a
worked 2-fiber example, before touching `model.py`:
- The exact tied block structure for Q/K/V/out and the MLP `fc1/fc2` on the g=12
  fiber (including the reflection action — tap-cycle reversal + axis transpose +
  fiber relabel; ×12 makes reflection first-class, so get its signs/permutations
  explicit here).
- Whether the bias must additionally tie across the head axis within a fiber-orbit
  (not just across board orbits) — the survey flagged this as the specific
  under-specification.
- The **group-norm** spec (normalize over the fiber structure, affine tied per
  `C_orbit`) and the **orbit-tied `LayerScale`** spec.
- The **typed-lift stem** map: 13 scalar planes (11 kept + 2 fork) → trivial rep
  copied into all 12 slots; 12 axis-planes → 4 copies of the 3-dim axis-permutation
  (S3) module lifted by the axis 3-cycle/transposition.
- **Exit check:** a paper proof (or tiny numpy prototype) that the composed block
  is exactly D6-equivariant, before committing to `model.py`.

**Phase 3b — implementation.**
- **Tied HexNodeConv:** `w_base:(7, C_in_orbit, 12, C_out_orbit)` + a precomputed
  tap-permute/fiber-relabel index-gather; **eager** full `(7,C_in,C_out)` weight
  generation inside `forward` (so it survives the `SERVE_HALF` deepcopy
  `inference.py:384` and CUDA-graph capture). Add the **dense-weight cache keyed on
  base-param `_version`** so frozen-serve weights regenerate once, not per forward.
  Recompute fan-in init on the orbit basis. (Reference GEMM `model.py:370-374` and
  the fp16 Triton conv/conv+LN kernels consume the generated weight unchanged.)
- **Group-norm + orbit-tied `LayerScale`** at all 8+ sites: `stem_ln 607`,
  ConvBlock `ln1/ln2 394-396`, AttnBlock `ln1/ln2 509-511`, `ln_final 623`,
  `LayerScale.gamma 377-385`.
- **Equivariant attention** per the 3a derivation — tied Q/K/V/out + `fc1/fc2`
  (pure `nn.Linear` reparam, invisible to the Triton attn kernel); tokens as a
  learned `(NUM_TOKENS, C_orbit)` broadcast over fibers.
- **Group-pool heads:** fiber-mean before every reduction (`_value_input
  1012-1016`, `_pooled 938-942` gains a fiber-mean); per-cell heads
  (`_policy_logits 1018-1027`, `_cell_q_logits 1029-1038`) get a fiber-invariant
  read so the per-cell logit lands in the trivial rep. Prune dead tokens 6&7
  (`NUM_TOKENS` 8→6).
- **Drop `HEXFIELD_CONV_FP8` for v1** (avoids the `id(weight)` cache conflict at
  `_triton_conv.py:359-374`; revisit later with a base-param `_version` re-key).
- **Checkpoint self-description:** persist `group_order`, `C_orbit`,
  `feature_width`, the orbit LUT, and the bias `free_rows` reduction in the
  checkpoint **`meta`/`extra`** (geometric LUTs also as buffers). Teach **both**
  arch inferers to read `meta` first (`model.infer_net_kwargs_from_state_dict
  551-577` AND the dashboard's `debug_infer.py:_infer_hexfield_arch 931-959`),
  shape-inference as fallback; add `in_channels` from `stem.weight.shape[1]`.
- **Param-grouping predicates** (`plugin.py:38`, `trainer.py:261-266`) updated for
  the new names (tied base params, `free_rows`, group-norm affine).
- **Gate — the equivariance unit test** (this test is the *sole* correctness
  guarantee that makes the augmentation-free expand sound): `f(g·board)` equals
  `g·f(policy)` and `f(g·board).value == f(board).value` to fp32 tolerance for all
  12 `g`. Plus: reference-path ≡ Triton-path parity preserved; grads reach every
  base param; a from-scratch net trains a few steps with no NaN.

### Phase 4 — BC-prefit (the go/no-go) + fix width (D5)
The first proof the whole stack learns. No fallback if it fails.
- Fix width per D5 (match the current c=128 param budget under ×12 tying).
- Cold-start **BC-prefit** from the HF corpus at 25 planes, augmentation-free
  (identity expand). No warm start bridges 15→25 / full-conv→tied.
- **Gate (mirrors the model spec's M3):** AMP run, no NaN; held-out top-1 within
  tolerance of the current hexfield BC reference on the same split;
  `value_ece ≤ 0.08`; probe harness online (minus `probe_d6_kl`, which is
  identically zero for an equivariant net — strip it and its consumers,
  `prefit.py:226-268`). **This is the project go/no-go**: if the 25-plane
  equivariant net doesn't at least match the current BC reference, stop and
  diagnose (features vs equivariance vs width) before any soak.

### Phase 5 — Eval re-anchoring & serve
- Re-anchor per D4: `permanent_anchors = ()`; pool pinned on Strix + SealBot +
  new-bot self-anchors (`multistage_eval` Stage D tolerates an empty
  permanent-anchor list). Do **not** point the arena at 15-plane `main5`/`main6`.
- Confirm serve: `SERVE_HALF` scalar-head fp32 casts survive (`model.py:998,1007`);
  tied-weight generation + the dense-weight cache behave under
  `torch.compile(dynamic=True)` (`inference.py:445`) and CUDA-graph capture
  (`448-453`); measure the `|support|×18` window-feature gather on the serve hot
  path (scatter-from-`entries()` + a separate openness pass if the per-cell gather
  bites).
- New `.service` env file (`HEXFIELD_GROUP_ORDER=12`, `HEXFIELD_C_ORBIT=…`, trunk,
  radius) + new run toml (copy main_11's shape; arch via service env per
  convention, selfplay/training/eval via toml).
- **Gate:** a full multi-stage eval runs against Strix/SealBot; serve parity gate
  (3e-3) holds; live serve throughput within budget.

### Phase 6 — Soak & the equivariance A/B
- Self-play soak (a few unattended epochs): sane entropy/length/calibration bands.
- **A/B (per D6):** the equivariant net vs a matched-width `HEXFIELD_GROUP_ORDER=1`
  ablation from the same BC-prefit start. If a fair ablation needs augmentation
  (D6a) that path is ported guarded to the ablation build; otherwise (D6b) gate on
  **absolute strength vs Strix/SealBot** and accept the coarser attribution.
- **Gate / kill criterion:** if the equivariant net does not at least match the
  ablation (or the Strix/SealBot bar under D6b) over the A/B window, the ×12
  equivariance is not paying for itself — fall back to `GROUP_ORDER=1` (which still
  ships graded features + orbit-tied bias) and reconsider ×6 or augmentation.

---

## 4. Cross-cutting concerns (apply across phases)

- **Two silent-corruption traps** (both from the survey): (1) name-based param
  grouping (`plugin.py:38`, `trainer.py:261-266`) must track every rename
  (tied base params, `free_rows`, group-norm affine); (2) two duplicated arch
  inferers must both learn the new arch — **fix once** by carrying arch in `meta`
  and reading it first.
- **Checkpoint meta is load-bearing** — `group_order`, `C_orbit`, `feature_width`,
  orbit LUT, bias reduction. Foreign loaders (eval_arena, dashboard) rebuild from
  it; without it a 25-plane / tied-weight checkpoint is silently mis-built.
- **Cold start only** — no warm start bridges 15→25 planes or full-conv→tied.
- **Tests to add/regenerate:** featurizer parity (25 planes), all-12-D6 expand +
  axis-permutation, orbit-tied-bias equivariance micro-test, the full-net
  equivariance test (Phase 4 gate), reference-vs-Triton parity, checkpoint
  round-trip + meta-driven foreign rebuild.

## 5. What NOT to port into `hexfield_eq` (vs the hexfield source it's copied from)
Because this is a fresh package, "strip" = "omit when copying hexfield's modules."
The line refs below are the hexfield sources whose logic is dropped/replaced.
- Binary hot/win planes + `HOT_MIN_PLACEMENTS` gate (constants Py+Rust,
  `features.py`/`features.rs`, `replay_expand.rs`, `shards.py`/`window.py`/
  `samples.py`, `selfplay.py:312-325`). Replaced by graded `window_features`.
- The 4 hot/win shard CSR columns (new schema, no hot/win columns).
- **The entire D6 augmentation machinery is never ported** (`D6_SIZE`/`_aug_seed`/
  the draw `trainer.py:615-617`; `samples.py` `symmetry` threading `243-392`;
  `features.py:transform_facts 246-266`; the Rust `sym`/`transform_facts` path in
  `replay_expand.rs`) — the expand path is identity-only. The `probe_d6_kl` probe
  (`prefit.py:226-268`) is omitted (identically zero for an equivariant net).
- fp8 conv path omitted for v1 (revisit with a base-param `_version` cache re-key).
- Dead tokens 6 & 7 (`NUM_TOKENS` 8→6).
- The live `hexfield` package keeps its legacy arch-inference maps + anchors
  untouched; `hexfield_eq` carries only its own.

## 6. Risk register (ranked)
1. **Equivariant attention derivation (Phase 3a)** — top risk, and now on the
   critical path with **no Tier-1 fallback bot** (the straight-to-Tier-2 choice
   removed the safety net). Mitigate: paper/numpy proof of block equivariance
   before any `model.py` code; ×6 remains a *technical* fallback (D1 chose ×12, but
   a Phase-3a finding that ×12 reflection is intractable can still retreat to ×6 +
   reflection augmentation).
2. **BC-prefit go/no-go (Phase 4)** — with no intermediate bot, a prefit miss is a
   project-level stop. Diagnose along three axes (features / equivariance / width).
3. **LayerNorm/LayerScale equivariance** — pervasive (8+ sites) but mechanical once
   3a fixes the affine-tying rule.
4. **Train-time WindowStore plumbing** — resolved by the small public engine
   constructor (Phase 1.1).
5. **Silent param-grouping / arch-inferer drift** — resolved by meta-first arch +
   predicate updates (both fixed in Phase 3b).
6. **Serve perf of the per-cell window gather** — measure in Phase 5; scatter form
   as fallback.
7. **Cross-arch eval** — resolved by D4 (re-anchor on Strix/SealBot/self).
8. **A/B attribution** — with no Tier-1 control, the equivariance verdict leans on
   a `GROUP_ORDER=1` ablation or an absolute Strix/SealBot bar (D6).

## 7. References
- Survey anchors throughout are current-tree (`main_9-fastrow-strip`).
- Related: `docs/specs/hexfield_model_spec.md` (v1 model spec), the (abandoned)
  main_12 global-pooling plan (the *other*, arch-additive proposal — a
  precise map of the same cross-arch/persistence seams), and the next-bot design
  doc §2.3/§2.4/§3/§4.1 this plan implements.
