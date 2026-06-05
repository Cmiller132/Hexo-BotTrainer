# hexgnn sparse-graph speed spec — efficient re-representation, not deletion

**Goal:** carry the **same information the model uses today, at lower cost**, so the
GNN forward (memory-bound on edge-scale scatter/gather, ~60% of self-play wall) and
the Rust featurizer (O(edges), ~38%) both get cheaper. Throughput: kill the
0%-idle GPU gaps and raise pos/s. We **measure how far the model's view moves**
(policy KL / value Δ on real positions) at every gate.

Scope: a **moderate rewrite of the hexgnn lineage only** — forked Rust crate
`packages/hexgnn/rust/` → `hexo_models._rust.hexgnn` (the halted hexgt module is
untouched). **WS0 is done and gated green** (byte-identical fork). **The build is
ON HOLD** pending owner approval of this design + the visits option (1/2/3 below).

> **v3 — reconciled with the owner's authoritative design.** This adopts the
> representation in **`docs/analysis/NEW_BASIC_MODEL_DESIGNS.md` design #4 (the
> hex-GNN)**, which directly answers the stones+candidates / richer-edges redesign:
> **direction-typed hex-adjacency edges (6-direction one-hot + same-owner flag)** as
> the "useful edges" (NOT window-line edges — simpler and edges stay ≤6/node),
> tactical signals folded into **per-node scalars** (near-complete-line me/opp/
> contested), value via **mean+max pool + global scalars**. **OPEN RECONCILIATION
> (needs owner confirm):** direction-typed edges with per-direction message weights
> are **D6 via AUGMENTATION**, NOT D6-invariant by construction — this **drops** the
> current hexgnn's exact-invariance guarantee (the gated exact equivariance test
> would no longer pass; we'd switch to a D6-equivariance/augmentation regime, or
> tie message weights across the 6-direction D6 orbit to keep exact equivariance at
> more cost). See §4. **Revised acceptance bar: ≥50 pos/s @512 visits, full-game
> measured.**
>
> NOTE: I could not locate a *newly pushed* plan beyond this doc on any branch/PR
> (branch HEAD = our spec commit; main = merge of our work; PR #4 body/comments
> empty). Treating NEW_BASIC_MODEL_DESIGNS design #4 as authoritative; **please
> confirm that is the plan (or point me to it) and pick the D6 approach** before I
> build.

---

## 0. Non-negotiable: TSS is fully kept, and is separable from the GNN graph

Verified: `threats.rs:3` — TSS operates on the engine **WindowStore**, "no
graph/feature construction, no network." `tactical_cells()` / `min_hitting_set()`
read `state.board().windows()` directly. The whole TSS stack (WindowStore threat
index, tactical injection at expansion, phase-aware hitting-set HARD WIN/LOSS leaf
overrides, tactical move-selection guard) reads the WindowStore, **not** the GNN
graph. **Removing window *nodes* from the GNN graph does not touch TSS at all** —
TSS never read the GNN window nodes; it reads the WindowStore.

The ONE coupling: TSS injects tactical cells as tree children whose **priors come
from the GNN eval**, so those cells must remain candidates. **Invariant T0:** the
candidate set always ⊇ `active-window empties ∪ tactical_cells(state)`. The **full
TSS test suite is a hard gate at every phase** (injection-additive, hitting-set
override, move-guard, two-stone defense, VCF).

---

## 1. Where the cost is (measured, td128/gnn3, active=512, visits=512)

- Wall: **GNN forward (GPU) ~60%**, Rust MCTS+featurize ~38%, py glue ~1.5%.
- Forward op mix: matmul only ~13-26%; the rest is **elementwise + scatter/gather
  over edge-scale tensors** ⇒ **memory-bandwidth-bound on EDGES** (compile recovers
  2.2-3x of the eager launch overhead).
- Per-leaf graph grows ~7x opening→ply60: nodes 222→1569, **edges 1499→11420**,
  candidates 215→1507. **Edges are the enemy.**

Edge classes today (`candidates.rs:312-377`), directed:
| class | construction | midgame share (≈) |
|---|---|--:|
| **ADJACENCY** | stones+candidates within hex-dist 1 (~6/node) | ~55% |
| **CONTEXT** | SIDE hub ↔ EVERY node (`2·(N-1)`) | ~27% |
| CANDIDATE_/STONE_WINDOW | window-token hub ↔ member cells | ~13% |
| RECENCY | consecutive stones | ~3% |
| WINDOW NODES | count-3/4/5 tokens (extra nodes, not edges) | ~5-13% of nodes |

---

## 2. CHOSEN DESIGN (owner-directed): stones + candidates only, no window nodes, richer edges

**Node set = placed stones + candidate cells. No window tokens. No SIDE node.**
One uniform candidate rule, no phase gating.

### 2.1 Drop the SIDE node — pool instead (justified)
The SIDE node's only jobs were (a) carry global scalars (phase one-hot, move
number, own/opp stone counts) and (b) be the CONTEXT hub. Both are removable:
- (a) **Broadcast the global scalars onto every node's feature vector** (they are
  D6-invariant counts/phase — adding them to stones+candidates is exact and cheap).
- (b) The value head already has a **PMA global pool** over all nodes; with the SIDE
  node gone the value readout is **PMA-only** (`value_head_use_side=False`, already
  supported + tested). The policy/opp heads are per-candidate and never needed SIDE.

⇒ **The entire CONTEXT edge class (`2·(N-1)`, ~27% of midgame edges) and the SIDE
node disappear.** Behavioral cost: the value head loses the SIDE-token readout but
keeps the PMA pool + the global scalars now on every node — measured by closeness.

### 2.2 Remove window tokens as NODES; fold their info into features + edges
Window nodes were a hub to avoid same-axis cliques. We remove the hub and deliver
the same tactical relationship two ways:

**(i) Richer NODE features (already mostly present).** Candidate features already
summarize window context: `F_CAND_OWN_WIN{3,4,5}`, `F_CAND_OPP_WIN{3,4,5}`,
`F_CAND_NWIN_{OWN,OPP}`, `F_CAND_COMPLETE_{OWN,OPP}`, `F_CAND_WIN_NOW_OWN`,
`F_CAND_OPP_THREAT`. So a candidate already *knows* "how many own/opp windows of
each count pass through me." Add the analogous **per-stone** window-count features
(stones currently lack them) so a stone knows its own line involvement. This keeps
the per-cell window summary the window node used to provide.

**(ii) Richer, more useful EDGES — DIRECTION-TYPED hex-adjacency (per design #4).**
Per the owner's authoritative design, the "useful edges" are **pure hex-adjacency
edges (stone/candidate to its ≤6 hex neighbors) carrying a 6-dim one-hot of which
of the six axial directions the edge points**, plus an optional 1-dim
"both-endpoints-same-owner" flag. **Direction-typed edges are what let the GNN
distinguish lines from blobs** — a run of same-direction edges *is* a line, so the
4-round message passing detects near-complete lines and threats without any window
hub. This keeps edges at **≤6 per node (|E| ≤ 3|V|)** — the structural efficiency
that the window hub + CONTEXT fan-out destroyed. The window-membership counts a
candidate needs are already in its node features (2.2(i)); direction edges supply
the line geometry. (This REPLACES the window-line-edge idea from v2, which was more
complex and added edges.)

**D6 — OPEN RECONCILIATION (owner picks):** a 6-direction one-hot is **not**
D6-invariant; under a D6 element the 6 channels permute. With **per-direction
message weights** (design #4) the model is therefore **D6-equivariant only up to
augmentation** — the exact by-construction equivariance test (currently green) would
no longer hold; we adopt D6 **augmentation** (the 12 transforms permute coords +
direction channels) ± test-time averaging, matching design #4. **Alternative
(keeps exact equivariance):** tie the message weights across the 6-direction D6
orbit (a group-equivariant message function) so rotating the input permutes the
output exactly — more code, no augmentation. **Owner: augmentation (simple, per
your doc) or equivariant weight-tying (exact, more cost)?** Either way the D6 gate
becomes an *equivariance/augmentation* test, not the current exact-invariance one.

### 2.3 One uniform candidate rule (no phase gating)
`candidates = active-window empties (A) ∪ tactical_cells (T0) ∪ radius-n(open-line)
filler (B)`, with a **single radius n = 2** (measured: n=3→n=2 cut edges/leaf 30%
and raised pos/s +47%, and the documented n2-vs-n3 strong-move coverage gap is
~1%). Dead cells still dropped (`has_open_window`). No move-number gate.

### 2.4 What each edge class becomes
| was | now |
|---|---|
| CONTEXT (SIDE↔all) | **removed** (pool + per-node global scalars) |
| STONE_WINDOW + CANDIDATE_WINDOW (hub) | **removed** (window counts already in node features) |
| WINDOW nodes | **removed** (info → per-node tactical scalars) |
| ADJACENCY | **kept + enriched**: 6-direction one-hot + same-owner flag (design #4); shrinks ~30% via radius 3→2 |
| RECENCY | kept (or fold into node recency feature) |

Net: edges collapse to **≤6/node hex-adjacency** (|E| ≤ 3|V|) — no CONTEXT, no
window hub, no window nodes — with direction types making each edge more useful.

---

## 3. Estimated win (per ply band; re-measured at the gate)

| ply band | nodes/leaf now→new | edges/leaf now→new | cut |
|---|---|---|--:|
| opening (<10) | 222 → ~150 | 1,499 → ~700-800 | ~50% |
| early-mid (10-30) | ~600 → ~420 | ~4,000 → ~2,000-2,400 | ~45% |
| midgame (30-60) | 1,569 → ~1,050 | 11,420 → ~5,500-6,500 | ~45-50% |

Drivers: CONTEXT removed (~27% midgame) + window-hub removed (~13%) + radius 3→2
shrinks ADJACENCY (~30% of the ~55% adjacency share) ; line edges add back a small,
bounded set (count-3/4/5 windows only). Window nodes removed (~5-13% of nodes).
**Direct measurement in hand:** n=3→n=2 alone cut edges/leaf 1972→1370 (−30%) and
raised pos/s 28.0→41.2 (+47%, td96/gnn2 @512 active=512 no-PCR).

**Projected pos/s @512 visits (td96/gnn2, opening-region):** ~28 → ~50-60 from the
representation change, + Rust-opt/pipeline (~1.3x) → **~65-75 opening / ~35-50
full-game.** Honest: still ~2-2.5x; **≥100 @512 full-game also needs visits≈128-192
or further structural loss** (surfaced as the option below).

---

## 4. Behavioral-closeness metric (hard gate — TARGET, not guarantee here)

Removing window nodes is a **bigger representational change** than the rejected
edge-cap design, so closeness is a **measured target**, not a guaranteed exact
match. On a corpus of ~512 real positions across ply bands, run CURRENT vs NEW
representation through the same fixed weights and compare heads:
- **policy / opp_policy**: per-position `KL(softmax_current ‖ softmax_new)` over the
  shared candidate set; report mean + p95.
- **value**: `|E[value_current] − E[value_new]|` (decoded scalar); mean + p95.

**Gate (initial):** mean KL < 0.10, p95 KL < 0.25, mean |Δv| < 0.04, p95 < 0.08.
(Looser than an exact-tier change, reflecting the honest representational move; the
owner can tighten.) We report the actual numbers so we know exactly how far the
model moved, and can fall back (e.g. keep window nodes) if it moves too far.

---

## 4.5 D6 = EXACT invariance via tied steerable edges (owner decision B)

**Answering "how would keeping D6-invariant work, and the downsides?"**

**The subtlety.** Direction-typed edges want a per-direction transform. Our node
features are **invariant scalars** (owner/count/distance — they do NOT transform
under D6). Exact D6-equivariance constrains the 6 direction weights by
`W_{σ_g(d)} = ρ(g) W_d ρ(g)^{-1}`. With invariant features `ρ(g)=I`, and since D6
acts **transitively** on the 6 hex directions, this forces `W_0=…=W_5` — i.e. tying
to a single matrix would make the message **direction-agnostic** (blob = line).
**So discrimination requires a small hidden component that transforms under D6.**
(Group-lifting — 6/12 oriented copies of the whole feature — would also work but
multiplies compute; rejected per the owner.)

**The construction (tied weights + one steerable channel-set, ~zero cost).**
Each node carries, beside its invariant scalars `s ∈ R^c`, a tiny **steerable
2nd-moment feature** `T ∈ Sym(2)` (a 2×2 symmetric tensor = 3 numbers) that
transforms as `T → R_g T R_gᵀ` under a D6 element `g` (acting on the hex plane as
`R_g ∈ O(2)`; reflections included — Hexo has no chirality so they're free). Let
`e_0…e_5` be the fixed hex unit directions (`R_g e_d = e_{σ_g(d)}`). The message
layer (NO per-direction matrices):
- **scalars** update with the existing tied/shared weights using only invariant
  inputs (incl. the invariants of `T`): `s_v ← LN(s_v + MLP([s_v, Σ_u W·s_u,
  tr T_v, det T_v]))`.
- **steerable** accumulates direction structure: `T_v ← Σ_{u→v} γ(s_u,s_v,attr) ·
  (e_d e_dᵀ)`, where `γ` is a learned **scalar** gate on invariant features and
  `e_d e_dᵀ` is a precomputed constant 2×2 (no params). Optional channel-mixing
  `T ← Σ_k a_k T_k` acts on the channel index (commutes with `R_g`).
- **readout (heads)** uses only **O(2)-invariants** of `T` (`tr`, `det`, eigenvalue
  gap) plus `s` → output is exactly invariant.

**Why it's invariant.** Every learned weight touches invariant scalars or the
channel index (commuting with `R_g`); the only geometry is the constant `e_d`,
which transforms correctly; the readout takes O(2)-invariant contractions. So
rotating/reflecting the input rotates every `T` and permutes directions
consistently → `s` and all invariants of `T` are unchanged → **bitwise-equal output
for all 12 elements** (the existing exact equivariance gate stays, same tolerance).

**Why line-vs-blob survives (the whole point).** `T_v = Σ γ_u e_d e_dᵀ` is the
(gated) **2nd moment of the neighbor directions**. Collinear neighbors (a line) →
`T` is rank-1, aligned with the axis → large **anisotropy** (`λ_max−λ_min`); an
isotropic blob → `T ∝ I` → ~zero anisotropy. The invariant anisotropy/`det`
cleanly separates lines from blobs — and unlike a 1st-moment vector it does **not
cancel** for a straight line through the node (e_d and e_{d+3}=−e_d give the same
`e_d e_dᵀ`). Deeper layers read these invariants, so "I'm on a line" propagates.

**Cost (~zero vs untied).** Per edge: one scalar gate `γ` (same MLP cost as today)
× a constant 2×2 outer product accumulated into `T` — O(E·3) adds, negligible
beside the O(E·d²) message matmul. Channels added: 3 (one `T`) to a few. **No
6×/12× lifting.** Forward FLOPs ≈ untied-direction-weights; fewer params (tied).

**Downsides beyond code complexity (flagged honestly):**
1. **Expressivity restriction (real).** The model can use direction only through
   O(2)-equivariant features — i.e. **axis-alignment / anisotropy**, not arbitrary
   per-absolute-direction feature maps. For Hexo (a line game) the tactically
   relevant direction signal *is* axis-alignment, so the match is good — but a
   fully untied model could in principle learn richer direction-specific maps (at
   the cost of needing augmentation and losing exactness). Mitigate with a few `T`
   channels (more 2nd-moment capacity) if a closeness/strength gap shows.
2. **Reflection-invariance** (full D6 tie): mirror images are indistinguishable —
   free for Hexo (no chirality).
3. **Numerics:** use **squared/polynomial invariants** (`tr`, `det`, `‖·‖²`), not a
   raw `sqrt` eigen-gap, to keep gradients smooth near `T≈0`.
4. Negligible extra activation memory (the few steerable channels).

## 5. Rust optimization (the 38%)

1. **Single live-cell pass**: replace per-candidate `has_open_window` O(18) rescan
   with one sweep building a `HashSet<cell>` of live cells; O(windows·6+candidates).
2. **Static-topology reuse** across a position's leaves (board differs by few
   stones): cache the live-cell set + active-window enumeration keyed by board
   occupancy, reuse across that position's leaf builds.
3. **Dedicated featurize threadpool** so `build_graph`/featurize don't contend with
   `select_leaf_batch`'s rayon over 512 roots; raise rayon grain for tiny graphs.
4. Zero-copy buffer output + pinned scatter unchanged (parity-preserving).

## 6. Pipeline (kill the 0% GPU gaps)

Already present: select↔eval prefetch (`mcts.rs:440-545`), featurize↔forward
double-buffer (`mcts_eval.rs:146-216`), eval cache. Add:
1. **Deepen** `HEXGNN_EVAL_PIPELINE_DEPTH` 2→3/4 so the featurizer queues ahead.
2. **Verify/force GIL release** around `_host_to_device` + the forward
   (`inference.py:224-260`) so the GIL-free Rust featurizer overlaps the CUDA launch.
3. **Pinned-host ring buffer** instead of per-call `pin_memory()`.
4. Optional later: CUDA graphs (bucketed shapes) + fused relational-message scatter.

---

## 7. Phased build + gates (each phase: commit via clone, report numbers)

- **WS0 — fork + rebuild (DONE, green):** `hexo_models._rust.hexgnn`, parity/D6/
  model/value/losses/self-play all pass on the byte-identical fork.
- **WS1 — new representation (this doc):** stones+candidates nodes, drop SIDE +
  CONTEXT, remove window nodes, add line edges + per-stone window features, single
  radius=2. Changes BOTH halves (`candidates.rs`/`features.rs` Rust + the Python
  featurizer + `architecture.py` value-readout to PMA-only + edge_attr width).
  **Gate:** featurizer parity (Rust↔Python on the NEW layout), D6 (all 12), **full
  TSS suite**, **closeness metric within gate**, edges/pos per §3, pos/s up.
- **WS2 — Rust opt (§5):** featurize ms down, parity green, pos/s up.
- **WS3 — pipeline (§6):** GPU util saturated, pos/s up.
- **FINAL gate:** **≥50 pos/s @512 visits no-PCR on FULL-GAME self-play** (revised
  owner bar; report opening AND full-game), all quality gates green, then HOLD for
  launch.

Quality gates EVERY phase: featurizer parity (<1e-6 on the active layout), D6 (12
elements), the **full TSS suite**, the **closeness metric**, shard sanity (λ=0 hard
targets, recorded rows, opp-mask).

---

## 8. Rejected alternative (documented per owner request): phase-gated radius + hub cap

A prior design kept window nodes and used: a **phase-gated radius** (n=2 until move
30, else n=3), a CONTEXT-hub fan-out cap to {candidates, windows, recent-64 stones},
and an adjacency-near guard. **Rejected by the owner** ("I don't like the phase
gating; rather simplify by removing/simplifying windows than have this kind of
phase gating"). It was more behavior-preserving (≈exact CONTEXT-analytic option)
but structurally more complex and yielded a smaller edge cut (~30-40%). The chosen
design above is structurally simpler (fewer node types, uniform rule) and cuts more
(~45-50%), at the cost of a measured (not guaranteed) behavioral move.

---

## 9. Open decisions for the owner

**Revised acceptance bar (owner): ≥50 pos/s @512 visits, FULL-GAME measured.**
This is within reach of the rewrite: the pure-adjacency direction-typed graph
(|E| ≤ 3|V|, no CONTEXT/window edges) is a bigger edge cut than v2, so ~35-50
full-game @512 from the representation alone, + Rust-opt/pipeline (~1.3x) →
**~45-60 full-game @512** — clears 50 if the deeper cuts land. I will report
**opening AND full-game** numbers honestly at the gate.

**Two decisions needed before I build:**
1. **Confirm the design source** — adopt NEW_BASIC_MODEL_DESIGNS design #4
   (direction-typed adjacency, per-node tactical scalars, mean+max value pool), or
   point me to the plan you pushed.
2. **D6 approach** — augmentation (simple, per your doc) vs equivariant
   weight-tying (exact, more cost). This changes the D6 gate.
3. **Visits** (throughput vs search): visits=512 targets the 50-bar directly;
   visits=192 would comfortably exceed it (~100+) with a thinner search.

**Build remains on hold** until the owner confirms (1)+(2) and picks visits.
