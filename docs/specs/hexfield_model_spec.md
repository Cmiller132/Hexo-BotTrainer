# hexfield — model & lineage specification (v1, synthesis)

Status: PROPOSED — synthesized 2026-06-12 from a 5-design competition (lenses: minimalist,
search-first, training-first, geometry-first, systems-first) judged by a 3-judge panel
(soundness / simplicity / systems). Base skeleton = the unanimous winner (minimalist); grafts and
fixes below were each confirmed by ≥2 judges. Competition artifacts: `scripts/_wf_nm_brief.md`
(envelope), `scripts/_wf_nm_design_*.md` (the five designs). Review pass 2026-06-12 (ground-up
vs `docs/intro_to_hexo.md` + code-only verification of inherited claims): editorial/provenance
fixes applied in place; decision points §12.5–7 added and resolved same day (plain conv blocks;
pooled-cell value/aux input adopted; standing-win planes adopted, F = 15); §12.8 records the
owner's exploration-knob quarantine (FPU noise-zeroing + root policy temperature broken in the
trusted lineage — production defaults off, as-built in parity mode, numbers-gated).

Working name: **hexfield** (two designs converged on it independently; trivially renameable now).

Conventions: `N` = board nodes in one position's support set; `S = N + 8` (8 summary tokens);
`C = 96` channels; `F = 15` features; `B` = rows per forward; action id = packed u32
`((q + 2^15) << 16) | (r + 2^15)` (universal across engine/Rust/Python).

---

## 0. Thesis

The model's domain is the *field the stones induce* — the engine-true support set
(stones ∪ full legal set ∪ 1-ring halo) on the native hex lattice — with direction-typed 7-tap
convolutions for local reasoning (dense_cnn's trusted operation with the square-grid carrier
removed) and restnet-style global attention carrying 8 bidirectional summary tokens. One geometric
law builds the domain; one norm (LayerNorm); one relative-position bias table with a closed-form
index; one memory rule (a pair budget); one reply wire-format back to Rust (dense_cnn's proven two-key
contract byte-identical, plus one additive optional key, §5.2), so the PUCT search consumes
hexfield evaluations exactly as it consumes dense_cnn's. Every engine-legal cell carries a policy logit, so coverage loss is
structurally impossible. All code is greenfield — this is not a fork; existing lineages are
semantic references and test oracles only.

Search stance (owner directive 2026-06-12): the proven dense_cnn/restnet search semantics —
including the continuous scheduler, the performance backbone — are the inherited foundation, but
hexfield is **allowed to diverge where the divergence is well-reasoned and documented**. Four
designed search improvements (§5.4) are default behavior, not optional knobs, and two inherited
exploration mechanisms are quarantined off by default (§5.1 — owner-flagged 2026-06-12 as broken
in the trusted lineage). A single test-only `search_parity_mode` switch disables all four
divergences AND sets the quarantined knobs to their as-built dense_cnn behavior, so the
differential harness can still pin the core rewrite against the trusted implementation; it is not
a production configuration.

---

## 1. Input representation

### 1.1 Support set (the one geometric law)

Ground truth is the engine, never re-derived geometry:

1. `stones` = occupied cells; `legal` = engine legal set (empty ∧ hex-dist ≤ 8 of any stone;
   ply 0 ⇒ forced `{(0,0)}`).
2. `core = stones ∪ legal`; `halo` = cells hex-adjacent to `core`, not in `core`. Halo carries
   features, **never logits**.
3. `support = core ∪ halo`.

Geometric identities (each a property test, not a construction step): `core` = union of radius-8
disks around stones (always connected — every placement lands within 8 of an existing stone);
`halo` = exactly the distance-9 shell; one multi-source BFS of depth 9 from the stones yields the
support, the halo, and the `dist_to_stone` feature in one pass.

**Node order (layout contract):** per row, segments `[ legal | stones | halo ]`, each segment
ascending by packed action id (= ascending `(q, r)`; the packing preserves signed sort order).
**Legal-prefix property:** the legal nodes of row g are exactly slots `[0, L_g)` — the payload
ships one legal count per row instead of index arrays, and priors return positionally over the
prefix.

Edge cases: ply 0 ⇒ support = origin + 6 halo neighbors (7 nodes, 1 legal); `dist_to_stone` := 0
everywhere on that one state. Terminal states are never evaluated (tree backs up engine outcomes);
the payload path tolerates a zero-legal row.

Scale anchors: 1 stone → 271 nodes; mid-game ≈ 600–1500; long spread games ≈ 3k. **No cap exists
anywhere** (a cap would be a crop).

### 1.2 Node features (F = 15: the 13 trusted planes + 2 standing-win planes)

Indices 0–12: same semantics as dense_cnn's planes (`dense_cnn_restnet/constants.py:29-41`,
`dense_cnn/rust/src/encoding.rs`); only index 11 is redefined (crop-center distance →
distance-to-nearest-stone). Indices 13–14: engine-exact standing-win planes (adopted 2026-06-12,
§12.7). f32 at train time, f16 on the wire (all values exactly or near-exactly
representable; the dense f16-transport gate rationale applies).

| idx | name | value at node v |
|-----|------|-----------------|
| 0 | own_stone | 1 if stone owned by side-to-move |
| 1 | opp_stone | 1 if opponent stone |
| 2 | empty | 1 − f0 − f1 (legal and halo cells) |
| 3 | legal | 1 if v ∈ engine legal set |
| 4 | phase_second | constant 1 iff phase == SecondStone |
| 5 | first_stone | 1 at this turn's first placement cell (SecondStone only) |
| 6 | player_colour | constant 1 iff side-to-move == player0 |
| 7 | own_recency | max over own placements at v of `1/(1 + latest_idx − placement_idx)` |
| 8 | opp_recency | same for opponent |
| 9 | opp_hot | 1 if v is an EMPTY cell of an opponent single-colour window with **count ≥ 4**; gated `placements_made ≥ 7` (encoding.rs:226-266 exactly) |
| 10 | own_hot | same for own windows |
| 11 | dist_to_stone | `min_hex_dist(v, stones) / 8`; stones → 0; legal ∈ (0,1]; halo = 1.125 exactly |
| 12 | opp_last_turn | 1 at the cells of the opponent's most recent full turn |
| 13 | opp_win_now | 1 at the single empty cell of each opponent count-5 active window (their standing win — a placement there wins them the game) |
| 14 | own_win_now | same for own count-5 active windows (the side to move's win-in-1 cells) |

- Hot threshold stays at the trusted **count ≥ 4** — deliberately identical to the TSS threat
  definition (`threats_shared.rs:21-23`): one "threat window" concept repo-wide.
- Standing-win planes (13/14) are the same engine window scan at **count == 5** — a count-5
  active window has exactly one empty; the plane marks it, window owner picks own vs opp. No
  placements gate needed (count-5 windows cannot exist before placement 9). Rationale: the ≥4
  hot planes conflate "completable within a turn" with "completable with ONE placement"; the
  4-vs-5 grade is the win-now / forced-block / race-tempo distinction the rules turn on at
  SecondStone (B = 1), and these channels hand it over instead of making conv re-derive it from
  raw stones.
- No halo flag: halo ≡ `empty=1 ∧ legal=0`.
- Parked (post-v1, gated experiment): graded per-axis line-potential channels replacing the binary
  hot planes (geometry-first design §2.2-2.3) — game-theoretically motivated (axis identity +
  count grade), but the competition's largest representational risk surface; if ever tried, it is
  an A/B at the BC-prefit gate with a pre-committed one-line reversion.

---

## 2. Trunk

Interleave **`C C C A C C A C A`** — 9 layers, 6 conv residual blocks, 3 attention blocks.
Per-layer rationale: stem + C1–C3 give receptive radius 7 ≥ the 5-step span of a 6-window before
any global mixing; A4 introduces the tokens; C5–C6 re-localize with token-informed features; A7 is
the hub's second round (bidirectional aggregation needs ≥ 2); C8 sharpens locally; A9 ends the
trunk so cells and tokens are maximally fresh for the heads.

Trunk state: cells `(B, Npad, C)` + tokens `(B, 8, C)`. C blocks update cells only; A blocks run
on the joint sequence `[8 tokens ; cells]` (length S_pad) and split back. Tokens initialize once
from a learned `(8, C)` parameter at A4 and are **carried through** (never re-initialized).

Provenance (review pass 2026-06-12, against code): the summary tokens are a **hexfield-designed
addition, not a restnet port**. restnet's attention runs over board cells only, and its
value/aux heads read a fixed spatial `Conv1x1 → Linear(1681→64)` reduction (`ValueReduction`)
whose own docstring justifies the spatial Linear by the crop being center-relative — a rationale
with no variable-N analog. The tokens are the N-invariant replacement for that readout; the pooled-cell
insurance for the value path is adopted (§12.6, wiring in §2.4).

### 2.1 HexNodeConv (the direction-typed 7-tap primitive)

Weight `(7, C_in, C_out)` + bias. Tap 0 = center; taps 1–6 = the fixed direction order
`D = [(1,0), (0,1), (−1,1), (−1,0), (0,−1), (1,−1)]` — the rotate60 orbit of (1,0):
`rot60(D[i]) = D[(i+1) mod 6]`, `reflect(D[i]) = D[5−i]` (used by tests only).

Forward: gather `(B, Npad, 7, C_in)` via `nbr_idx` (tap 0 = self; missing neighbor → index of an
appended all-zeros row = conv zero-padding semantics), reshape, **one GEMM** with `(7·C_in, C_out)`.
No cuDNN convs exist in this model — the 925 ms-per-novel-shape autotune hazard is structurally
absent. `nbr_idx` is built once per batch and shared by all 15 conv applications.

This is mathematically dense_cnn's hex conv family (the masked 3×3's seven surviving taps, one
weight matrix per relative direction, shared everywhere). **Anchored executably**: a test embeds a
random support into a 41×41 grid, copies the 7 direction matrices tap-for-tap into dense_cnn's
masked `HexConv2d`, and asserts equality at the support cells (fp64) — the trusted implementation
as oracle, never as dependency.

### 2.2 Conv residual block (×6)

Post-activation residual — restnet's plain `ResidualBlock` form, **not** dense_cnn's gated
3-conv `GatedResBlock` (the gate is one of the few structural deltas between the two lineages;
§12.5) — with LN in place of BN:

```
y = ReLU(LN1(Conv1(x)))        Conv1, Conv2: HexNodeConv C→C
y = LN2(Conv2(y))
x = ReLU(x + y)
```

**Norm = LayerNorm, and only LayerNorm, everywhere** (stem, conv blocks, attention, final norm).
The knob is deleted, not defaulted: on variable-N node sets, batch statistics couple gradients
across samples keyed on game length, break train==eval parity, and lag at the BC→RL transition;
LN has no running stats, no fold/fuse machinery, and makes micro-bucket gradient accumulation
*exactly* equal to monolithic batches (a tested theorem, §6.3). Disclosure: **both** trusted
lineages run BN in their conv blocks (model1's gated blocks and restnet's R blocks alike), so
LN-in-convs deviates from each. Empirical tripwire: the BC-prefit
gate (M3) — gross undertraining vs the restnet BC reference triggers the documented contingency
(masked BatchNorm1d over valid nodes). Two of five designers independently chose LN; both judges
who examined the norm question corroborated.

**Init:** trunc_normal(0.02) on Linears; conv weights PyTorch default (fan_in = 7·C_in);
LN weight 1 / bias 0 — **except the residual-closing parameters, which are zero-initialized**
(each conv block's LN2 gain; each attention block's out_proj and MLP fc2 weights), so every
residual branch is the identity at step 0. Plus a 500-step linear LR warmup on fresh
initializations only (resumes skip it). Both are flag-reversible conditioning insurance.

### 2.3 Attention block (×3)

Pre-norm restnet transformer block — exact *block* semantics of the trusted implementation
(pre-norm MHSA + GELU MLP at ratio 2, 4 heads, sdpa/materialized dual impl). The bias table
below is hexfield's own design; restnet's is a full `((2H−1)(2W−1), heads)` exact-offset table
on the fixed 41×41 crop, which has no variable-geometry analog:

```
seq = seq + Attn(LN1(seq))     # RelPos MHSA, 4 heads, head_dim 24, scale 1/√24
seq = seq + MLP(LN2(seq))      # Linear C→2C, GELU, Linear 2C→C
```

Two numerically identical attention impls share parameters: `sdpa` (production; bias as additive
attn_mask) and `materialized` (test oracle) — the restnet dual-impl pattern.

**Relative-position bias — ONE shared learned table, shape (237, 4 heads):**

| rows | meaning |
|------|---------|
| 0–216 | exact axial offset `(dq, dr)`, hex-dist ≤ 8 (the 217 offsets of the radius-8 disk) |
| 217–224 | **on-win-axis** ring buckets, hex-dist 9–16 (offset collinear with a win axis: `dq==0 ∨ dr==0 ∨ dq+dr==0`) |
| 225–232 | **off-axis** ring buckets, hex-dist 9–16 |
| 233 | far bucket, hex-dist ≥ 17 |
| 234 / 235 / 236 | (query=cell, key=token) / (query=token, key=cell) / (query=token, key=token) |

Index function (query i, key j; `Δ = (q_j − q_i, r_j − r_i)`, `d = max(|dq|,|dr|,|dq+dr|)`):
exact LUT for d ≤ 8; `217 + 8·(off_axis) + (d − 9)` for 9 ≤ d ≤ 16; 233 beyond; token rows/cols by
class. The axis split is exactly D6-invariant (rotations 3-cycle the win axes, reflections
transpose them) and preserves the lattice's anisotropy where overlapping-window chains live
(+8 rows/head over pure rings, zero extra runtime). The table is **shared across all 3 A layers**
(per-row variable geometry makes the gathered `(B, 4, S, S)` bias the dominant memory term; sharing
means one alive tensor, not three). The integer pair-index `(B, S_pad, S_pad)` is a pure function
of node coords, computed once per batch on-GPU and reused by all 3 layers.

Tokens sit at sequence slots 0–7 with no board position. Pad-cell KEY columns get additive
**−3.0e4** (finite in fp16 — closes restnet's documented −1e9→−inf saturation hazard). Token keys
are **never masked**, so every attention row has ≥ 8 live keys and a fully-masked softmax row is
structurally impossible. Bias table zero-init.

### 2.4 Final norm and head taps

After A9: one shared `LN_final` over the joint sequence. Cells feed the policy heads; tokens split
**T0–T1 → main value, T2–T3 → STV + moves-left aux, T4–T7 uncommitted hub capacity**.

Both value-style readouts also receive the **masked mean-pool of cells** (adopted 2026-06-12,
§12.6): `pooled` = the elementwise mean of the row's N real cells' `LN_final` vectors (padding
excluded by mask), computed once per row and shared by both readouts. Value head input =
concat(T0, T1, pooled) = 288; aux reduction input = concat(T2, T3, pooled) = 288. Rationale:
both trusted lineages read value from the **cell field**, and with the zero-initialized
attention residuals (§2.2) the tokens are position-independent constants at step 0 — the pooled
path gives the value/aux heads board signal from the first gradient step, while the tokens
remain the learned, content-weighted refinement. The per-layer token-attention-mass probe
(§6.4) meters whether the hub earns its keep.

---

## 3. Heads & losses

Target semantics are exact ports of the verified restnet constructions (restnet
`losses.py`/`samples.py` imported in tests as oracles), re-keyed from crop flats to support nodes.
Loss reduction is always **mean over rows, never over nodes** (the variable-N bug rule).

| head | tap | architecture | output | target | mask | w |
|------|-----|--------------|--------|--------|------|---|
| policy | cells | HexNodeConv(C→C) → ReLU → Linear(C→1), **logits gathered at legal nodes** | (L_g,) per row | MCTS visit weights, action-id → legal-prefix slot | structural (support = legal set; zero coverage loss asserted) | 1.0 |
| opp_policy | cells | same shape, separate params, **legal nodes only** | (L_g,) | next opponent decision's visit policy, projected onto THIS row's legal set, renormalized | rows with no target / masked-from-fast / zero projected mass contribute 0 (`allow_zero_rows`); `opp_target_coverage_mean` telemetered | 0.25 |
| value | T0,T1 + pooled cells | concat(288) → Linear(288→96) → ReLU → Linear(96→65) — private MLP | (65,) | hard z ∈ {−1,+1} → adjacent-bin soft target | — (truncated games never written, §5.1) | 1.0 |
| stvalue_{2,6,16} | T2,T3 + pooled cells | shared concat(288) → Linear(288→96) → ReLU; per-horizon Linear(96→65) | (65,)×3 | even-offset EMA of future root values, decay (m−1)/(m+1) | per-row horizon mask | 0.1 each |
| moves_left | T2,T3 + pooled cells | Linear(96→65) on the shared aux reduction | (65,) | `clamp(remaining, 0, 512)/512 → [−1,1]` → 65-bin (cap applied at expansion, raw scalar in shards) | −1 sentinel mask | 0.1 |

- **Pathway separation (heads_v3 natively):** main value reads tokens {0,1} + the shared pooled
  cell vector through its own MLP; non-stationary aux targets read tokens {2,3} + the same
  pooled vector through a separate reduction. Separation by token *and* readout (the pooled
  vector is a shared *input*, not a shared bottleneck — each reduction owns its weights).
- Policy CE = segment soft cross-entropy over each row's legal prefix (scatter-logsumexp, fp32);
  target mass off the legal set is a hard error. No −1e9 fill exists in the loss path at all —
  legality masking is structural because the logit support *is* the legal set.
- 65 bins `linspace(−1,1)`; decode = softmax expectation clamped to [−1,1]. Hard-z only in v1;
  any future value-target-family change requires a value-head LR ramp or fresh head (the
  relocation-shock procedure rule, encoded in config docs).
- No spatial ownership/win-window head (owner-skipped).
- moves_left's −1 sentinel/mask path exists almost solely for the Phase-B legacy adapter
  (truncated games are never written and the BC corpus is decisive-only) — kept deliberately;
  not dead code.
- Total = `1.0·policy + 1.0·value + 0.25·opp + 0.1·Σstv + 0.1·moves_left`.

---

## 4. Symmetry — D6 by training-time augmentation

12 transforms about the **origin** (forced-opening anchor; no crop center exists):
`rot60(q,r) = (−r, q+r)`, `reflect(q,r) = (q, −q−r)`, indices 6–11 = reflect-then-rotate. Per
training row, one symmetry is drawn (pipeline-supplied) and applied to **all stored coordinate
facts** (stones, history, first_stone, hot cells, win-now cells, last-turn cells, policy /
opp-policy action ids);
the support set, node order, neighbor table, features, and bias indices are then *rebuilt from the
transformed facts*. Nothing else transforms — no weight permutation, no table permutation
(the trusted augmentation-not-invariance approach). Construction commutes with D6 (legality and
adjacency are hex-distance-based), so augmentation is exact for 100% of rows with zero drops — the
crop lineage's spill/drop machinery is deleted, not ported. Validation runs identity symmetry.
Inference is always identity.

Tests: support/feature/target construction commutes with all 12 transforms up to the canonical
re-sort permutation; direction algebra `rot60(D[i]) = D[(i+1)%6]`, `reflect(D[i]) = D[5−i]`;
σ∘σ⁻¹ = id.

---

## 5. Search integration

### 5.1 Provenance: from-scratch rewrite, own crate

Search is **written from scratch** in hexfield's own Rust crate (owner-locked: not a fork;
extraction of the live lineage's search was considered by one design and rejected — it performs
source surgery on the live main_4 lineage mid-run and bets on a permissive reading of the
from-scratch lock).

**Packaging (build blast-radius discipline):** `packages/hexfield` ships its **own cdylib**
(`hexfield._rust`, own maturin crate) — it is NOT a `#[path]` submodule of `hexo_models._rust`,
whose single-crate design makes every rebuild change search semantics for all lineages at once.
Building hexfield can therefore never touch the live run's `.so`. `threats_shared` is consumed via
a `#[path]` file-include from the own crate (the hexgnn-precedent pattern; works without same-crate
compilation), with a cross-crate drift parity test; promotion of threats_shared to a tiny rlib is a
later, owner-scheduled option. All verification builds run in a separate dev venv
(`/root/.venvs/hexfield-dev`), never the live `hexgt-build` venv — the live supervisor relaunches
processes between epochs and would pick up a replaced `.so` from disk.

**Baseline search semantics (the as-built trusted contract, dense_cnn as semantic reference;
`search_parity_mode` reproduces exactly this):** batched PUCT
with virtual loss; prior-sorted lazy edge materialization; nucleus widening
`policy_mass 0.95 / max_children 96 / min_children 2` (production main_4 config values; the Rust
code-default for max_children is 32); FPU + `root_fpu_zero_under_noise` (**quarantined**, below);
Dirichlet root noise (total_alpha / fraction); root policy temperature incl. the early/halflife
schedule and the reused-root application (the main1 reuse bug is a named regression test;
**quarantined**, below);
tree/subtree reuse; move selection by per-ply temperature; **continuous scheduler**
(`run_continuous` semantics: per-game slots, virtual_batch_size, flush_target, active_root_limit,
on_move callback with the existing payload keys) plus the lockstep `search` driver; **PCR**
(full/fast coin per ply, fast = pcr_fast_visits, no noise, not recorded); **policy-init openings**
(truncated-exponential ply draw, raw-prior sampling at policy_init_temperature); forced playouts
for Full roots **including KataGo policy-target pruning of the exported visit-policy targets**
(most-visited child never pruned; the played move comes from the un-pruned distribution — a
production semantic the visit-count parity assertions alone would not catch);
**the exact `mix_seed` hash and stream ids 0–5 adopted as a written contract**;
**TSS toggleable** via the landed `tss_enabled` key at the three proven sites (expansion injection,
leaf override, root guard) — with the call-site crop filter deleted: every tactical cell is always
in-vocabulary, so injection is total by construction. main_4's frozen-win override (C3) is **not
ported** — crop-frozen wins are unrepresentable here. The length-decay knees (C1) ride along with
the 1:1-ported replay machinery and stay **dormant by default**: marathons driven by
value/exploration dynamics remain possible even without the crop, and the dormant lever costs
nothing. `drop_truncated_rows` behavior IS adopted: truncated games' rows are never written.

**Exploration-knob quarantine (owner directive 2026-06-12):** two of the inherited exploration
mechanisms — `root_fpu_zero_under_noise` and the root-policy-temperature machinery (the
early/halflife schedule + the reused-root application) — are **known to be broken / not working
properly in the trusted lineage**, whose exploration numbers are nonetheless good. They are
therefore part of the as-built parity surface but NOT trusted semantics. The differential
harness reproduces dense_cnn's *actual* behavior, bugs included — that is what pins a rewrite —
while hexfield **production defaults ship plain FPU reduction with no noise-zeroing, and root
policy temperature 1.0 (schedule off; reused and fresh roots identically untouched)**. Enabling
either in production requires, first, a trace-level characterization of what the dense
implementation actually does versus what the knob claims, and second, a dedicated matched-visits
A/B. The exploration-health bar is the **measured numbers** (§6.4 exploration panel, M9 band
gate), never the mechanism list. Dirichlet root noise, PCR, policy-init openings, nucleus
widening, and per-ply move-*selection* temperature are not implicated and ship as inherited.

**Internal seam:** the new crate structures its search around a one-trait evaluator boundary
(`LeafEvaluator`: unique states → evaluations) with generic dedup/cache/order-restoration behind
it — costless in a greenfield crate, and it makes the differential harness below trivial to wire.

**Differential parity gate (the rewrite's safety net):** run hexfield search and dense_cnn search
on the same ≥100-position corpus — **constrained (and asserted) to positions whose full legal set
lies inside dense_cnn's radius-20 crop**, since dense's candidate vocabulary is in-crop legal and
any clipped position diverges by construction, not by bug — same seeds, hexfield in
`search_parity_mode`, with a deterministic stub evaluator (priors/values = pure hash of
state_hash/action_id). Identical PUCT constants + identical seed streams + identical priors ⇒
assert **identical visit counts, chosen moves, and exported (forced-playout-pruned) visit-policy
targets**, for lockstep AND continuous, with
**trace-level assertions** (per-move noise draws, widening counts, move-class sequences) so a
failure localizes the divergent semantic instead of forcing a blind bisect. TSS on/off coverage and
the reused-root temperature regression are part of this gate.

**Eval cache:** `HashMap<StateHash, Arc<Evaluation>>`, key = `hexo_utils::hash_state` (pure engine
hash, encoder-independent), bounded at **262,144 entries** (production main_4's value — full-legal
priors cost ~4–12 KB/state, so the legacy 1M code-constant would mean 5–10 GB of host RAM here),
Arc-shared, in-flight dedup of duplicate misses.
Stored-prior truncation (a search-first proposal) is **deferred**: its tree-exactness proof fails at
noised roots (Dirichlet alpha is drawn over the full candidate list), so it may not ship until that
analysis is redone; host RAM is not currently scarce.

### 5.1b The continuous scheduler is the performance backbone

restnet's continuous scheduler (per-game slots, per-slot state machines, single flush queue,
select↔eval overlap, games refilling as they finish) is the best measured throughput design in the
repo and is retained as-is in semantics. hexfield's performance work layers **around** it, in the
evaluator, where it cannot perturb scheduling semantics: the static-shape packer (§5.3), a
**featurize↔forward overlap pipeline** (within-call chunk pipelining, the in-repo-proven pattern:
while the GPU forwards packed shape j, the CPU featurizes/packs/uploads shape j+1 of the same
flush — featurization is O(N) per leaf here, not a fixed plane-stamp, so hiding it matters;
cross-flush double-buffering would require future leaf sets to exist early, which is a scheduler
property, and is at most an M8-measured option), per-shape persistent staging buffers (CUDA-graph
static
addresses), and the fp16 adopt-and-gate. Scheduler knob defaults (visits 512, c_puct 1.5 —
i.e. §5.4.3's c_init, active_root_limit, virtual_batch_size, flush_target) mirror production
main_4.

### 5.2 Evaluator payload ABI

Request (Rust → Python), one dict per flush; CSR flat-concat; **rows pre-sorted by support size
descending in Rust** (stable by request index) so Python grouping is contiguous slicing; the dedup
slot-map restores caller order on reply.

| key | dtype | length | meaning |
|-----|-------|--------|---------|
| `abi` | int | 1 | schema version = 1 (fail-loud drift guard) |
| `shape` | tuple | (B, total_nodes) | cross-checks every buffer |
| `node_feats` | f16 zero-copy buffer | T × 15 | features, node-major, §1.1 segment order per row |
| `node_qr` | i16 bytes | T × 2 | axial coords (Python builds bias indices on-GPU from these) |
| `node_row_offsets` | i64 tuple | B + 1 | node CSR |
| `nbr` | u16 bytes | T × 6 | row-LOCAL neighbor index per direction D; sentinel 0xFFFF; fail-loud if any row N > 65,534 |
| `legal_counts` | i32 bytes | B | L_g per row (legal-prefix property — no index arrays) |

≈ 46 bytes/node. Rust keeps the per-row sorted `Vec<PackedCoord>` action ids; they never cross the
boundary.

Reply (Python → Rust) — **superset of dense_cnn's ABI** (the two base keys are byte-identical;
one optional key added):

```
{ "values_bytes":  f32 × B,                    # binned-value expectation, clamped [−1, 1]
  "priors_bytes":  f32 × Σ L_g,                # per-legal softmax, positional over the prefix
  "moves_left_bytes"?: f32 × B }               # present when request_moves_left is set —
                                               # DEFAULT ON (§5.4.4 consumes it); off only in
                                               # search_parity_mode or under ML auto-disable
```

Rust zips positionally with retained action ids, validates (finite, non-negative, positive mass,
exact byte counts; action-id uniqueness is an invariant of Rust's own retained per-row lists —
ids never cross the boundary), descending-sorts, normalizes → `Evaluation{value, priors}`. Because the
vocabulary is the full legal set, `legal_action_count == priors.len()` always. The PUCT tree
consumes `(action_id, prior)` pairs opaquely — everything downstream of the evaluator is untouched
by hexfield's representation. moves_left decode = median-of-bins mapped to decisions [0, 512],
clamped (consumed by §5.4.4). Serve-forward surface, stated explicitly: policy + value always; the
aux reduction + moves-left top when requested (~4% of per-node MACs); the **opp-policy head never
executes at serve** (train-only), saving ~7% of conv MACs vs the train forward.

### 5.3 Python evaluator: sort-and-pack into static shapes

Kernel-shape stability is the **evaluator's** property, not the scheduler's. Per flush:

1. One H2D copy per array (frombuffer → cuda, non_blocking).
2. **Deterministic packing** (rows arrive size-sorted): greedily fill a fixed static shape table
   `(S_c, B)` ∈ {(384,64), (512,48), (768,32), (1024,24), (1536,16), (2048,8), (3072,4)}. The
   first five shapes hold a uniform 24,576 cells/forward; the two large-S shapes deliberately
   shrink to 16,384 / 12,288 cells so the `(B, 4, S, S)` fp16 bias transient stays ≤ ≈ 305 MB —
   the **inference pair ceiling** `B·S_pad² ≤ 3.8e7`, distinct from §6.3's 2.0e7 *training*
   collate budget. Smaller rows fill tail slots of larger-shape batches; the rare > 3072 tail
   falls back to ceil-to-256 with B = the largest batch under the same 3.8e7 inference ceiling
   (assert-logged). A 54-leaf
   staggered flush, a 256-leaf continuous flush, and a lockstep round all decompose into the same
   ≤ 7 shapes, so per-shape torch.compile / CUDA-graph economics survive variable N and the
   measured avg-batch-54 staggered-root pathology is absorbed here.
3. Scatter CSR → padded per shape; neighbor globalization on GPU (`sentinel → zero-row`).
4. Forward per shape (fp16 weights behind the restnet-style adopt-and-gate); pair-index built once,
   bias gathered from the shared table; worst transient bias ≈ 305 MB (shapes (1536,16) and
   (3072,4) tie).
5. Per-row prefix segment softmax (proven scatter pattern, fp32), values decoded + clamped.
6. Single D2H sync; reorder to caller order; emit bytes.

### 5.4 Designed search divergences (default behavior; owner-approved 2026-06-12)

hexfield's search diverges from the trusted lineage in exactly four documented ways. They are
default behavior, not production knobs. Internally each divergence has a test-only toggle (used by
the M10 lesion A/Bs and the M6 property gates); `search_parity_mode` is the composite that forces
all four off AND sets the §5.1 quarantined exploration knobs to as-built dense behavior, for the
differential harness. None of these are production configuration surface.

1. **LCB move selection on greedy paths.** Production self-play temperature-samples its played
   moves (halflife-decayed T with floor 0.1); "best move" selection only exists on greedy paths:
   PCR fast moves (~75% of decisions), lockstep / eval-ladder / arena play, and plies where
   effective T → 0. On those paths the move is chosen by lower-confidence-bound of Q —
   `Q − z·σ/√n`, z = 1.6, σ² from a per-edge sum-of-squares accumulator (a tree-stat schema
   addition, inert in parity mode), eligibility = visits ≥ max(8, 0.1·max_child_visits), fallback
   to max-visits when no child qualifies. Temperature-sampled Full moves are unchanged — KataGo's
   validated LCB Elo lives in greedy match play. Exploration and visit-distribution training
   targets are untouched.
2. **Early-stop overtake pruning, scoped by move class.** Unrestricted on greedy, *unrecorded*
   searches — PCR fast moves and eval/arena play (~43% of search compute at production PCR
   settings): stop when the remaining budget cannot change the selection **under the rule actually
   in force** (visit-overtake, AND where LCB is active, LCB winner == visit winner); the freed
   continuous-scheduler slot refills immediately. Where LCB is active this stop test is
   **conservative-heuristic, not exact** — remaining visits can still move Q/σ or make a new
   child LCB-eligible — so the M6 gate splits: exact chosen-move equality with LCB lesioned off
   (pure visit-overtake IS exact), statistical agreement with LCB on. On recorded Full roots the targets are
   temperature-sampled distributions, so stopping early changes the recorded target — there a
   conservative visit floor applies instead (no stop before 75% of budget), which captures most
   tail savings while protecting forced playouts and noise-driven exploration mass. Telemetry:
   early-stop fraction and mean visits-at-stop, by move class.
3. **Visit-scaled c_puct.** `c(n) = c_init + c_scale·log((n + c_base)/c_base)`, shipped at
   c_init 1.5, c_scale 0.45, c_base 500 — +0.04 at n=50 (small-n behavior preserved, checkable
   arithmetic), +0.32 at n=512. Rationale: fixed c_puct under-explores large subtrees and
   over-explores small ones, and Hexo's 300–800 branching makes the imbalance worse than in Go.
   Expectation note: static c_puct sweeps (1.1/1.5/2.0) measured as noise in this repo, and the
   dynamic schedule stays within roughly that band — this is the divergence most likely to lesion
   to ≈0 at M10; it ships with the deliberately small c_scale, and a null lesion result is not a
   failure.
4. **Moves-left utility — the stage-3 mechanism, adopted verbatim.** A selection-time per-edge
   PUCT bonus, never value/backup shaping:
   `U_ml(e) = − w_ml · g(Q_e) · tanh((M_e − M_node)/m_scale)`, shipped at w_ml = 0.03,
   m_scale = 32, g = the |Q| > 0.6 win-side-only gate (the bonus is identically zero near Q = 0 —
   no sign discontinuity exists). The delta-vs-sibling-baseline form is invariant to the head's
   absolute bias by construction (the documented failure mode of absolute moves-left readings).
   Tree mechanics: per-node `(ml_sum, ml_weight)` accumulated on real backups; terminals
   contribute exact path distance (off-by-one fix); head decode = median-of-bins mapped to
   decisions [0, 512]. PCR fast searches are steered identically. Source:
   the (removed) stage-3 moves-left feasibility analysis — the mechanism was validated there; the only
   blocker was the flood-damaged legacy head, and hexfield's head trains on clean targets from BC
   prefit onward. Health: the stage-3 L0 metrics (conversion-zone within-game Spearman ≥ 0.6,
   [0,5) median-decode MAE ≤ 15, end-vs-mid pairwise ≥ 0.85, correct-sign sibling decrements) run
   as per-epoch monitoring with **auto-disable**; the stage-3 probe assets (64 squander + 39
   control positions) are reused as the nightly control-flip probe. Directly attacks the in-crop
   game-lengthening that survived main_4's armor.

Considered and **not** pursued: DAG/transposition-sharing search (the eval cache already dedupes
GPU work across transpositions and its measured hit rate is 1.2% at 512 visits — tree-stat sharing
buys little and carries known correctness hazards); stored-prior truncation (§11, refuted proof);
raising visits (the evals/position budget is an owner-level call, not a search-code lever).

### 5.5 Throughput & cost honesty

Per-eval MACs at serve ≈ `0.85M·N + 0.22M·S + 576·S²` (the opp-policy head never runs at serve;
the train forward is ≈ 0.91M·N) vs the dense serve-reference ≈ 2.9 G-MAC fixed:
≈ 0.3× at N=600, ≈ 0.5× at the N≈900 mid-game median, crossover ≈ N 1500, ≈ 2.9× at the 3k
marathon tail (which the crop could not represent at all). Honest projection: **0.6–1.2× dense
evaluator wall-clock at typical N** before packing/compile gains; the per-row bias materialization
is the #1 priced inefficiency (~10–15% of forward); FlexAttention (computing bias inline from
coords, enabling jagged batching) is the designated post-v1 escape hatch behind the repo's
≥10%-measured-win lever discipline. TRT is explicitly not pursued in v1 (dynamic-shape gather
graphs export fragilely; the measured 2.4–2.7× was on the fixed-shape dense model). The real cap
remains evals/position (visits × game length), unchanged. M8 measures before anyone believes a
number.

---

## 6. Data pipeline

### 6.1 Shard schema — `hexfield_compact_v1`

Compact-facts concept (one columnar `.npz` + JSON sidecar per game; raw representation-agnostic
facts; encoders expand at train read), with three hygiene changes vs the legacy layout:

- **No legal-id column.** Legality is closed-form from stones (`empty ∧ hexdist ≤ 8`;
  Opening ⇒ {(0,0)}); expansion derives it, and a CI invariant test pins the derivation to the
  engine's `write_legal_moves` on random **non-terminal** states (on terminal states the engine
  reports zero legal moves while the closed form does not; decision rows are never terminal).
  Removes the largest column and the only stored
  fact that could silently disagree with search.
- **Stones and history unified:** one column `(q i16, r i16, owner u8, placement_index u16)` IS the
  placement history; recency, opp_last_turn, and first_stone derive from it.
- `phase` stored u8 enum (no object/pickle column).

Kept per row: `turn_index i32`, `current_player u8`, `value f32`, `moves_left f32` (raw remaining;
−1 sentinel; cap applied at expansion), `first_present/first_q/first_r`, `stvalue (n,3) + mask`;
CSR: unified stones, `own_hot_qr` / `opp_hot_qr` and `own_win_qr` / `opp_win_qr` (stored —
window logic stays single-sourced in the engine-backed featurizer; the win columns are the
count==5 standing-win cells of §1.2), `pol_act/pol_w`, `opp_act/opp_w`. Sidecar JSON keeps the dashboard's
field set + `"lineage": "hexfield"`, `"schema": "hexfield_compact_v1"`.

**Legacy read adapter:** hexfield can read existing restnet compact-v1 shards by **ignoring their
stored legal_ids** (crop-restricted at the source) and deriving legality from stones — crop-clipped
marathon rows re-expand with full supports. Stored hot lists are raw engine coords (not
crop-clipped) and read as-is; legacy shards have no win-now columns, so the adapter derives them
from the stored stones via the same window scan, property-tested against the engine.

### 6.2 Expand-time featurization

Per row, in Python DataLoader workers: draw symmetry → transform facts → build support + BFS
distances → features → `nbr_idx` → targets, with strict fail-loud validation (finite, non-negative,
positive policy mass, policy-target support ⊆ legal set). **Python is the primary, debuggable train-time featurizer;
Rust is the serve-time featurizer; a fixture parity test (exact ints; floats compared in f32
**before** the f16 wire cast at ≤1e-6 — recency values quantize at ~1e-4 in f16, so a post-cast
comparison would need a loose, less diagnostic tolerance; node order exact) is the contract
between them.** (A Rust-only featurizer was considered and rejected: it puts
a WSL maturin rebuild into every feature-debug loop.)

### 6.3 Replay window and trainer

Port restnet's machinery semantics 1:1: policy-surprise frequency weighting materialized as row
duplication at finalize; mtime-ordered tapered shuffle window (keep 300k rows, taper 0.65,
md5-hash train/val split, `cp -p`-compatible); keep-prob subsample → permute → batch-aligned
shards; PCR row filtering at the
source (only Full-search rows written); truncated games never written. Optimizer identical: AdamW
lr 1e-3, wd 1e-4 on matrix weights only (no decay: biases, LN params, token inits, bias table), AMP
+ GradScaler, grad-clip 1.0, **no EMA in v1** (verified 2026-06-12: the restnet trainer has no EMA
machinery — there are no "restnet EMA semantics" to port; adding one is a recorded option, not a
port), strict-load checkpoints (no silent partial loads).

**Trainer deltas (the only three):**
1. **Pair-budget micro-bucket collate:** each nominal 32-row batch is sorted by N and split into
   micro-buckets under the rule `B_g · S_pad² ≤ 2.0e7` (bounds the dominant `(B,4,S,S)` bias
   transient at ≈ 160 MB in fp16 under autocast — ≈ 320 MB if the gather ever runs fp32; the one
   training memory knob). One optimizer step per 32-row batch via gradient
   accumulation with **step-global denominators** (per-head unmasked-row counts computed at
   collate), which under LN is *mathematically identical* to a monolithic batch — enforced by
   tests, not assumed.
2. **Pair-index reuse** across the 3 A blocks.
3. **Exactness test trio** as permanent regression gates: padded-batch vs single-row forward
   identity; micro-bucket accumulation ≡ monolithic gradients; train-mode ≡ eval-mode bit parity.

### 6.4 Per-epoch diagnostics contract

Adopted wholesale from the training-first design (the competition's highest-value graft — this
repo's collapses were all diagnosed forensically after the fact; these make ignition visible per
epoch for seconds of GPU):

- **Loss panel:** per-head unweighted CE + weighted total; `opp_target_coverage_mean`;
  masked fractions.
- **Optimization panel:** grad-norm pre-clip {mean, p95, max}, clip fraction, update-to-weight
  ratio, AMP scale, `nan_trips` (must be 0; a non-finite loss dumps batch provenance and raises).
- **Value calibration panel (validation):** `value_ece` (E[v]-decile reliability vs realized z),
  `value_optimism = mean(E[v]) − mean(z)` (the autopsy-endorsed scalar), extreme-bin mass,
  value-logit scale, CE split by outcome class and **by game-length quartile** (the recurring
  failure axis, made visible), moves-left MAE in decisions.
- **Gradient-interference panel (once per epoch):** per-head trunk grad norms + 
  `grad_cos(policy,value)` / `grad_cos(value,aux)` on one probe micro-bucket — direct measurement
  of the interference the token split is designed to bound.
- **Fixed-probe drift telemetry:** freeze ~1024 rows with realized outcomes at run start; forward
  them every epoch (identity symmetry, eval mode); report `probe_policy_kl_prev` (policy-churn
  meter — ignition shows here epochs before Elo), probe entropy, `probe_value_ece` vs the frozen
  real outcomes (longitudinally comparable), |ΔE[v]|, per-layer token-attention-mass (hub health),
  and a **D6-consistency probe**: mean policy KL between `f(T_s·x)` and `T_s·f(x)` on the probe set
  — for an augmentation-trained (non-equivariant) model, the direct meter of whether D6 training is
  working; reuses the probe forward at near-zero cost.
- **Search telemetry (per epoch):** LCB-override rate (LCB pick ≠ visit pick), early-stop fraction
  and mean visits-at-stop by move class, moves-left flip rate and auto-disable status.
- **Exploration panel (per epoch):** root prior entropy pre/post noise; root visit-policy entropy
  by move class; distinct-children-visited fraction and widened-children histogram at Full roots;
  played-move temperature trace by ply bucket; opening diversity (distinct ply ≤ 6 prefixes per
  100 games). These are the numbers the §5.1 quarantine decision holds the line on — exploration
  health is judged against production main_4's measured values, never by which mechanisms are
  enabled.
- **Data panel:** legal-set / support-size distributions, window stats.

---

## 7. Bootstrap

**Phase A — BC prefit from the HF corpus** (`timmyburn/hexo-bootstrap-corpus`, 6,902 decisive games
≈ 431k positions, raw move-lists), following the proven recipe shape: replay through `hexo_engine`
(legality/terminal validated), one-hot policy on the played move, hard-z from the engine winner,
STV masked (no search values), moves_left real; write production shards with the production writer;
KataGo shuffle; production trainer; D6 on; game-level train/val split; 3–5 passes.
Gates: held-out top-1 within 2 pts of the restnet BC reference on the same split;
`value_ece ≤ 0.08`; probe harness online; (LN tripwire lives here).

**Phase B — optional distillation accelerant:** read recent restnet self-play shards through the
legacy adapter (legality derived from stones — supports are crop-free even on crop-clipped rows;
the stored *visit policies* remain crop-limited at the source and are disclosed as such). Rows
tagged `source=legacy_shard`; 1–2 passes in prefit only; **never seeds the RL replay window** (the
on-policy buffer starts from fresh hexfield self-play only). The gold-standard variant (replay
stored placement history through the engine and re-encode natively) is documented as the converter
upgrade if Phase B earns its keep.

---

## 8. Code architecture

```
packages/hexfield/
  pyproject.toml                  # maturin; module hexfield._rust; deps: hexo_engine, hexo_utils,
                                  # hexo_train, hexo_runner, torch, numpy  (NOT hexo_models)
                                  # entry point [hexo_train.models] hexfield = hexfield.plugin
  python/hexfield/
    constants.py geometry.py      # F indices, bins, caps, D order, LUTs; hex math, D6, packing
    support.py features.py        # train-time truth: support build, BFS, features
    model.py                      # HexNodeConv, ConvBlock, AttnBlock (sdpa+materialized), net
    losses.py samples.py          # segment CE, 65-bin helpers; finalize targets (z/opp/STV/ML)
    shards.py replay.py           # hexfield_compact_v1 writer/reader + legacy adapter; window
    batching.py                   # pair-budget collate, packer shape table, nbr/global indices
    inference.py                  # HexfieldEvaluator: evaluate_payload ABI + direct inference
    selfplay.py trainer.py        # continuous-scheduler epoch driver; train hooks + probe.py
    evaluation.py checkpoints.py  # anchored-ladder eval; strict loader/saver
    player.py plugin.py           # runner/arena adapter; hexo_train plugin surface
  rust/src/
    lib.rs                        # #[pymodule] hexfield._rust (own cdylib)
    support.rs features.rs        # serve-time truth: support + features + sample facts
    payload.rs                    # ABI assembly (size-desc sort) + reply parse/validate/finalize
    tree.rs search.rs             # PUCT tree; lockstep + continuous, PCR, policy-init, seeds, TSS
    cache.rs                      # bounded state_hash-keyed eval cache + dedup (trait seam)
    threats_shared.rs             # #[path] include of hexo_models/rust/src/threats_shared.rs
configs/hexfield_main_9.toml      # [model] module = "hexfield.plugin"
scripts/_rebuild_hexfield.sh      # maturin develop --release (dev venv by default)
scripts/bootstrap_hexfield_hf.py
tests/test_hexfield_*.py
```

Links against shared infra only (`hexo_engine`, `hexo_utils`, `hexo_train`, `hexo_runner`,
threats_shared file-include). Zero runtime imports from dense_cnn/restnet/hexgt/hexgnn — those are
imported **in tests only, as executable oracles** (HexConv2d conv equivalence; losses/samples/replay
target math; sdpa-vs-materialized pattern).

### Test strategy (the contracts that keep two of anything honest)

1. Attention oracle: sdpa ≡ materialized (≤1e-4 fp32), incl. tokens + padding.
2. Conv oracle: gather-GEMM ≡ dense_cnn `HexConv2d` on embedded supports (fp64), incl.
   missing-neighbor zeros.
3. Rust↔Python featurizer parity: golden fixtures over ~50 random engine states; exact node order,
   features, nbr, legal counts, BFS distances.
4. Search differential vs dense_cnn: stub-evaluator visit-count equality, lockstep AND continuous,
   with trace-level assertions (noise draws, widening counts, move classes); TSS toggle; reused-root
   temperature regression; mix_seed golden vectors.
5. ABI golden: payload round-trip; byte-count and validation failure modes.
6. Shard tests: writer round-trip; legacy restnet fixture cross-read (legality derived ≡ engine);
   D6 commutation property suite.
7. Exactness trio: padded-vs-single-row; micro-bucket ≡ monolithic grads; train ≡ eval.
8. Loss masking: empty-opp rows, masked STV/ML, zero-denominator exact-0, target-hygiene raises.
9. Checkpoint strict-load (bidirectional key equality; mismatch raises).
10. e2e smoke: 4-game, 64-visit epoch through the real plugin on CPU.

---

## 9. Perf budget (paper figures; M8 measures)

Parameters (C=96, 4 heads, mlp_ratio 2) — **≈ 1,230,651 ≈ 1.23M**:

| component | params |
|---|---|
| stem (7-tap 15→96 + LN) | 10,368 |
| 6 conv blocks | 777,600 |
| 3 attention blocks (QKVO + MLP + LNs) | 224,352 |
| shared bias table 237×4 | 948 |
| tokens + final LN | 960 |
| policy + opp heads | 129,410 |
| value head (288→96→65, tokens + pooled cells) | 34,049 |
| aux reduction (288→96) + 4 tops (96→65) | 52,964 |

VRAM: training peak ≈ **< 1.5 GB** at batch 32 under the pair budget (bias ≤ 160 MB, conv gathers
≈ 0.5 GB, activations ≈ 0.3 GB, params+grads+Adam ≈ 25 MB); inference ≤ **≈ 0.5–0.7 GB** per packed
chunk (worst shape bias ≈ 305 MB). Both comfortable beside the live evaluator on the shared 12 GB
card. fp16: f16 wire transport; fp16 eval weights behind the adopt-and-gate; LN/softmax/CE in fp32.
torch.compile: optional, per static shape, correctness-gated (the model has no data-dependent
Python guards on the eval path by design).

---

## 10. Milestones (each gate blocks the next; GPU scheduling vs main_4 out of scope)

| # | deliverable | acceptance gate |
|---|---|---|
| M0 | geometry, support, features (Python) + property tests | BFS-9 ≡ support on random states; halo = dist-9 shell; ply-0 = 7 nodes; D6 commutation 12/12; standing-win planes ≡ engine window scan on random states |
| M1 | model.py forward/backward | param count == §9; sdpa ≡ materialized; conv oracle vs HexConv2d; exactness trio; pad-inertness; grads reach every param |
| M2 | losses, samples, shards, batching | target math == restnet oracles on fixtures; legacy-shard cross-read with derived legality; writer round-trip; pair-budget collate respected |
| M3 | BC prefit on HF corpus + probe harness | AMP run, no NaN; top-1 within 2 pts of restnet BC reference — **anchor honesty:** restnet is the workspace's *struggling* learner; if a model1 (dense_cnn) BC number on the same corpus/split exists or is cheap to produce, report it alongside as the more meaningful anchor; `value_ece ≤ 0.08`; probe npz per epoch; **LN-vs-BN tripwire decision point** (gated conv blocks, §11/§12.5, are the second contingency) |
| M4 | Rust support/features + sample facts | Rust↔Python parity fixtures exact |
| M5 | payload + lockstep search | ABI goldens; stub-evaluator visit parity vs dense_cnn lockstep (≥100 positions, exact, with traces) |
| M6 | continuous scheduler + PCR/policy-init/noise/TSS + the §5.4 divergences | stub parity for continuous in parity mode (move classes, chosen moves, visit counts, exported targets); seed-stream vectors; TSS toggle; reuse-root regression; **divergence property gates**: (all divergences off + §5.1 quarantined knobs as-built) ≡ `search_parity_mode` bit-for-bit; quarantine plumbing trace-verified (parity mode reproduces dense's as-built FPU-zeroing and root-temp behavior; production defaults ship both off); early-stop on≡off chosen-moves **exact** over ≥1k greedy searches with LCB lesioned off, and ≥99.9% agreement (every disagreement logged with its Q-gap) with LCB on (§5.4.2 — the LCB stop test is conservative-heuristic, not exact); LCB vs closed-form on synthetic visit/Q/σ tables incl. fallback; ML-utility sign/monotonicity and gate-zero-below-\|Q\|=0.6 properties |
| M7 | plugin + e2e | 4-game 64-visit epoch through hexo_train in the dev venv; artifacts/diagnostics/checkpoint round-trip; strict-load |
| M8 | perf calibration | measured evals/s vs dense at matched settings (floor: ≥ 0.8× restnet's continuous-scheduler pos/s on a mid-game mix; target: parity+); featurize↔forward overlap pipeline measured on/off; packer shape histograms + tail-occupancy / padded-cell fraction (documented fallback: half-B shape variants, only if measured to bite); fp16 gate; VRAM within §9; compile go/no-go. **Floor-miss path** (the §5.5 honest projection 0.6–1.2× overlaps the floor): at a measured 0.6–0.8×, the §11 FlexAttention lever and half-B variants become mandatory M8 follow-ups re-measured before M9, or the owner renegotiates the floor — M9 does not start below the floor without owner sign-off |
| M9 | self-play soak | 2–3 unattended epochs; sane entropy/length/calibration bands; **exploration numbers within band of the production main_4 reference at matched settings** (root visit entropy, distinct-children coverage, opening diversity, game-length distribution — land in band by tuning the working knobs: noise alpha/fraction, move-selection temperature, widening; do NOT enable quarantined mechanisms to chase it, §5.1); prefit-seeded bot ≥ smoke-parity vs its own BC checkpoint over 100 games; handoff doc |
| M10 | search-divergence lesion study (§5.4) | per-divergence lesion arena A/B at matched visits (turn one off, measure) — for attribution and constant tuning, not ship gates; moves-left auto-disable threshold calibrated here |

---

## 11. Deferred levers (recorded, not shipped)

- **Stored-prior truncation** (cache/tree RAM ~7–9×): the bit-exactness argument fails at noised
  roots (Dirichlet alpha spans the full candidate list); re-derive before any adoption.
- **FlexAttention** jagged batching (deletes bias materialization): adopt only on oracle
  equivalence + ≥10% measured win.
- **Graded per-axis line-potential features** (replaces binary hot): gated BC-prefit A/B with
  pre-committed reversion; largest representational risk in the competition.
- **threats_shared rlib promotion**: owner-scheduled; until then the file-include + drift test.
- **Masked-BN contingency**: only if the M3 LN tripwire fires.
- **Flat-CSR convs** (delete pad compute): only if M8 shows pad waste materially hurts.
- **Gated conv blocks** (dense_cnn's `GatedResBlock`: `x + main(x)·σ(gate(x))`, 3 convs/block,
  +~50% conv params/MACs): the most successful lineage's block form; the contingency if M3/M9
  show the plain-block trunk underlearning (§12.5).
- **D6-orbit-tied bias table**: tie the 217 exact-offset rows over D6 orbits (~40 free rows) for
  an exactly-symmetric long-range prior; adopt only if the §6.4 D6-consistency probe stays high
  late into training.

## 12. Owner decision points (everything else above is self-contained)

1. **Lineage name** — "hexfield" (rename now is free).
2. **LayerNorm-everywhere** — a deliberate deviation from the trusted dense_cnn BN recipe, chosen
   independently by two designers and corroborated by the judges, with the M3 tripwire +
   masked-BN contingency. Ratify or veto.
3. **Hot threshold count ≥ 4** (trusted semantics, = TSS threat definition) vs the brief's "≥ 3"
   wording. Spec says ≥ 4.
4. ~~Optional `moves_left_bytes` reply field — keep or delete?~~ RESOLVED 2026-06-12: kept; it is
   load-bearing for the moves-left-utility search lever (§5.4), which the owner directed the design
   to pursue as a gated improvement.
5. ~~**Conv-block form** — plain (restnet) vs gated 3-conv (dense_cnn)?~~ RESOLVED 2026-06-12:
   **plain** (owner: "plain only, we can try gated later"). The gate is not needed up front — its
   benefit is unproven off the crop, the attention interleave already provides what conv gating
   approximates least well (long-range context), and it costs +~50% conv params/MACs. It stays
   recorded in §11 solely as the pre-agreed response if the M3 BC gate misses badly.
6. ~~**Value/aux head input source** — tokens-only vs concat with pooled cells?~~ RESOLVED
   2026-06-12: **concat adopted** (owner). Value = concat(T0, T1, masked mean-pooled cells);
   aux = concat(T2, T3, same pooled vector). Both trusted lineages read value from the cell
   field, and with zero-init attention residuals a tokens-only readout is position-independent
   at step 0; the pooled path closes that cold start for ~28k params. Wiring in §2.4/§3; params
   in §9.
7. ~~**Standing-win planes?**~~ RESOLVED 2026-06-12: **adopted for v1, F = 15** (owner).
   `opp_win_now`/`own_win_now` = the single empty cell of each count-5 active window — the
   win-in-1 / forced-block / race-tempo grade the ≥4 hot planes erase, decisive at SecondStone
   (B = 1). Engine-exact, D6-safe, +1,344 stem params, +4 wire bytes/node. Spec updated
   throughout (§1.2, §4, §5.2, §6.1, §9, M0); the graded per-axis channels stay parked (§11) —
   these two planes are their safe slice.
8. **Exploration-knob quarantine** — RESOLVED as issued (owner directive 2026-06-12):
   `root_fpu_zero_under_noise` and the root-policy-temperature machinery are broken / not
   working properly in the trusted lineage, which performs well despite them. hexfield ships
   both OFF in production; `search_parity_mode` reproduces the as-built dense behavior for the
   differential harness; re-enabling either requires trace-level characterization + a
   matched-visits A/B; exploration health is gated on measured numbers (§6.4 panel, M9 band),
   not mechanism presence. Details in §5.1.
