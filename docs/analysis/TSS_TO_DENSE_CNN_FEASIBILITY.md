# Porting Threat-Space Search (TSS) from hexgt/hexgnn into dense_cnn — Feasibility Report

**Date:** 2026-06-09
**Status:** Analysis only. No code changed. Both live runs untouched.
**Scope:** Assess difficulty/scope of giving the dense_cnn (Model 1) lineage the
same TSS capability that the hexgt/hexgnn lineages already ship.

---

## TL;DR

Porting TSS to dense_cnn is **easier than expected**, because the part everyone
assumes is the hard part — the threat machinery itself — is **already
lineage-agnostic and already linked into dense_cnn's binary**.

- The `WindowStore` (incremental ">=4 window" threat index) lives in the
  **shared `hexo_engine`** crate, and dense_cnn **already calls it today**
  (`dense_cnn/rust/src/encoding.rs:196`, `sample_gen.rs:125`) to build its
  hot-cells input planes.
- The TSS analysis module `threats.rs` (tactical-cell collection, exact
  hitting-set, phase-aware verdict) imports **only** `hexo_engine::{HexCoord,
  HexoState, TurnPhase}` — **zero** graph/GNN/candidate-node types. It can be
  promoted to a shared module and consumed verbatim.
- dense_cnn, hexgt, and hexgnn all compile into the **same** native module
  `hexo_models._rust` via `#[path]` includes
  (`packages/hexo_models/rust/src/lib.rs:4-17`). Sharing `threats.rs` is a
  code-organization move, **not a cross-crate port** — same `maturin develop -m
  packages\hexo_models\Cargo.toml` rebuild that already exists.

So the cost is **not** "reimplement TSS." The cost is **adapter wiring** into
dense_cnn's own Rust MCTS at three hook sites, plus one genuine architectural
difference (the fixed 41×41 crop) that bounds — but does not block — the
expansion-injection piece.

**Recommended path:** ship the two crop-independent, no-retrain hooks first
(leaf override + move-selection guard) on top of the shared threats module.
That captures most of the tactical safety benefit for ~1–2 days of work. Full
parity including expansion injection is ~1 week.

---

## 1. Architecture gap: dense_cnn MCTS vs hexgt MCTS

Both lineages run a **Rust-native, batched PUCT MCTS** over the **same
`hexo_engine` `HexoState`**. The board substrate is identical; the divergence is
in how candidate moves are represented and bounded.

| Aspect | hexgt / hexgnn | dense_cnn (Model 1) |
|---|---|---|
| MCTS language | Rust | Rust (`dense_cnn/rust/src/mcts.rs`, `mcts_tree.rs`, `mcts_eval.rs`) |
| Board in search | `RustHexoState` owned by `RustSearch`; interior nodes via path-replay | Same pattern: `root_state: RustHexoState`, leaves reconstructed by replaying edge actions (`mcts_tree.rs:406-482`) |
| Per-cell occupancy in search | `state.board()` masks + `windows()` | **Same** — `board().get()`, `is_cell_empty()`, `occupied_cells()`, `windows()` all available inside the search (`encoding.rs:96-219`) |
| Move encoding | Graph candidate nodes → `PackedCoord` action ids | `PackedCoord` action ids, but **must be representable as a flat in the fixed 41×41 crop** centered on the action region (`encoding.rs:91`, `model1_crop_center` at `:261`) |
| Candidate set | All graph candidates staged; widening cap = top-p nucleus; **forced edges** can be injected out of prior order (`RustEdge.forced`) | Every **in-crop** legal move staged as a lazy `RustPriorCandidate`; widening cap = top-p nucleus (`max_eligible_children`, `mcts_tree.rs:125-127, 509-513`). **No `forced` edge concept for tactics** |
| Forced-visit mechanism | `edge.forced` → guaranteed first visit before PUCT/FPU (`hexgt mcts_tree.rs:513-520`) | Has `forced_playout_k` but that is **KataGo root-noise forced playouts** (`mcts_tree.rs:535-565`), a different mechanism — not per-edge tactical forcing |
| Leaf eval | byte payload → GNN forward → priors/value | byte payload → CNN forward → priors/value (`inference.py:267-335`, `mcts_eval.rs:339-395`) |
| Move selection | raw visit policy (guarded) for play; pruned policy for training (`hexgt mcts.rs:709-712`) | raw visit policy for play; pruned policy for training (`mcts.rs:599-614`) — **same split, already present** |

**Where each TSS hook attaches in dense_cnn:**

1. **Threat index** — *nothing to attach.* Already present via the shared engine
   `WindowStore`; dense_cnn reads it today.
2. **Expansion injection** — node construction / candidate staging in
   `mcts_tree.rs` (the `Shared`/`Owned` prior path around `shared_from_cache`,
   `:400`, and `materialize_next_candidate`, `:185`) plus the
   `select_or_materialize_edge` widening gate (`:509-545`).
3. **Leaf override** — the leaf-selection loop in `mcts.rs` where terminal /
   transposition leaves are backed up inline (`mcts.rs:522-528`); a third branch
   `threats::analyze(&state).verdict()` slots in alongside the terminal check.
4. **Move-selection guard** — the post-search visit-policy → action path
   (`mcts.rs:295-330` action selection; `:599-614` raw-vs-export policy split).
   The guard masks the **raw** weights only.

---

## 2. Reusability: is the TSS logic separable from the graph?

**Yes — almost completely.** This is the central finding.

- **`WindowStore`** (`hexo_engine/rust/src/tactics.rs:342-486`): pure board
  occupancy. State is `AHashMap<WindowKey, [u8;2]>` plus a `live_threats` index;
  it reads bitmasks and coordinate geometry, never a graph. It is shared engine
  infrastructure both lineages already link. **Reuse cost: zero.**

- **`threats.rs`** (`hexgt/rust/src/threats.rs`): its module header explicitly
  states "Pure board geometry over the engine's incremental `WindowStore`; no
  graph/feature construction, no network." Verified at the import line — it pulls
  in **only** `hexo_engine::{HexCoord, HexoState, TurnPhase}`. Contains:
  - `tactical_cells(state) -> Vec<HexCoord>` (own win-now ∪ all opponent
    >=4-window empties), `:174-199`
  - `min_hitting_set(sets, budget) -> Option<u8>` exact for B≤2, `:80-117`
  - `analyze(state) -> ThreatAnalysis` (phase-aware), `:119-161`
  - `verdict() -> Option<f32>` (HARD WIN +1 / HARD LOSS −1 / None), `:54-72`

  None of this references `PositionGraph`, candidate nodes, GNN features, or
  hexgt's tree types. It can be **lifted verbatim**.

- **Hitting-set** is plain set geometry over `Vec<HexCoord>` — exact pair-scan
  for budget ≤ 2, which is all the game phases require.

**What *is* graph-coupled** (and therefore *not* part of this port): the hexgt
**threat candidate features** (`features.rs:131-193`, schema v3 slots 30–31).
Those emit threat flags onto GNN candidate nodes. dense_cnn's analog is **dense
input planes**, and it already has weak threat planes ("hot cells",
`encoding.rs:196-210`). Enriching those is a *separate, retrain-gated* change,
not part of the search-time TSS port (see §4).

**Recommended structure:** promote `threats.rs` to
`packages/hexo_models/rust/src/threats.rs` (or `#[path]`-include it) and
`mod`-reference it from both `dense_cnn` and `hexgt` lib roots. Because all three
lineages are one compilation unit (`lib.rs:4-17`), this introduces **no new
crate, no new wheel, no new build command**.

---

## 3. Effort breakdown (concrete work items)

Sizing: **S** = <0.5 day, **M** = ~1–2 days, **L** = ~3+ days, each including
tests.

### 3.1 Shared threats module — **S**
Move/`#[path]`-include `threats.rs` so dense_cnn's MCTS can call it. Adjust
imports. No logic changes. Rebuild via existing maturin command. Risk: none.

### 3.2 Phase-aware hitting-set leaf override — **S→M**
At the dense_cnn leaf-selection branch (`mcts.rs:522-528`), after the terminal
and transposition checks, add:
```
else if let Some(v) = threats::analyze(&leaf_state).verdict() {
    backup(±v); skip network eval & node creation;
}
```
The leaf state is already reconstructed, occupancy is already queryable, and the
backup path already exists for terminal nodes. **Crop-independent** — a verdict
reads board occupancy + phase only; no child needs to be representable.
Risk: low. This alone removes a large class of 1-ply tactical blunders and
**costs no retraining** (it changes search values, not the policy target).

### 3.3 Tactical move-selection guard — **S→M**
At the raw-policy → action step (`mcts.rs:295-330`, gated before sampling at
`:609-614`), port `tactical_guard_weights` + `classify_root_move` (hexgt
`mcts.rs:913-992`): clone root, apply each root move, read the child's verdict,
then force proven wins / mask proven losses on the **raw** weights only. Leave
the **export/pruned** policy (training target, `:603-606`) untouched. The
raw-vs-export split dense_cnn needs **already exists**. **Crop-independent.**
Risk: low–moderate (must wire it into selection without touching the export
path; one careful diff). No retraining.

### 3.4 Tactical expansion injection — **M→L** (the hard part)
This is the only item with real friction, for two reasons:

1. **No `forced` edge mechanism.** dense_cnn's `RustEdge` has no tactical
   `forced` flag and materializes strictly by descending prior under the nucleus
   cap (`mcts_tree.rs:509-545`). To inject tactical cells you must port hexgt's
   `split_tactical` (`hexgt mcts_tree.rs:833-873`): add a `forced` flag, lift
   `max_eligible_children` **additively** by the count of tactical cells outside
   the nucleus, and guarantee each forced edge a first visit (hexgt
   `:513-520`). Interior threat nodes currently use the cheap shared-by-`Arc`
   prior path (`shared_from_cache`, `:400`); a node carrying threats must switch
   to an **`Owned`** mutable candidate list so the tactical cells can be ordered
   first — exactly the caveat hexgt's design notes call out
   (`docs/analysis/HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md §1.4`). The common
   (no-threat) path stays on the cheap shared route, so cost is paid only on the
   rare threatened node.

2. **The fixed 41×41 crop (genuine architectural difference).** A tactical cell
   is only injectable as a child if it maps to a flat in dense_cnn's crop
   (`encoding.rs:91, 261-296`). On hexgt's infinite candidate graph every
   tactical empty is representable; on dense_cnn an opponent >=4-window empty
   that falls **outside the crop has no child slot**.

   **Important nuance — this is not a new bug.** dense_cnn can only ever *play*
   in-crop moves (its policy is crop-relative). An out-of-crop covering move is
   already unplayable regardless of TSS. So "don't inject what can't be
   represented" is *consistent* with dense_cnn's existing action space, not a
   soundness regression introduced by TSS. The leaf override (§3.2) and guard
   (§3.3) still fire correctly because they read occupancy, not crop flats — they
   will *recognize* an out-of-crop forced loss even when injection can't *search*
   the answer. In practice length-6 threat windows are local and the crop is
   centered on the action region, so the overwhelming majority of tactical cells
   are in-crop; the work item is to **filter tactical cells to in-crop before
   injection** and accept the residual as a pre-existing crop limitation.

Risk: moderate. Touches the hottest tree code; needs careful tests for the
`Owned`-promotion path, additive cap, forced-first-visit, and the in-crop
filter. This is where most of the week goes.

### 3.5 (Optional, out of search scope) Threat input-plane enrichment — **M, retrain-gated**
Port the spirit of hexgt's v3 candidate features into richer dense planes
(win-now / must-answer flags beyond the current hot-cells planes,
`encoding.rs:196-210` ↔ `constants.py`/`constants.rs` plane indices, kept in
lock-step per CLAUDE.md). **Requires retraining** and a plane-schema bump, so it
**cannot be adopted by the live `dense_cnn_rl_main1` run** without a fresh model.
Not recommended as part of the initial port.

### Blockers
- **None hard.** The only structural constraint is the crop (3.4), and it
  degrades gracefully rather than blocking.
- **Operational constraint:** all of this requires a `hexo_models` native
  rebuild (`maturin develop -m packages\hexo_models\Cargo.toml --features
  python`). The **live `dense_cnn_rl_main1` run pins `PYTHONPATH` to the
  worktree packages** and would pick up a rebuilt `_rust` — so this must be
  developed/tested on a separate checkout (e.g. the `C:\Hexo-consolidate2`
  clone) and **not rebuilt against the running training tree**.

---

## 4. Lighter alternatives

1. **Override + guard only (RECOMMENDED first cut).** Ship §3.1 + §3.2 + §3.3,
   skip injection (§3.4). Crop-independent, no retraining, ~1–2 days. Captures:
   - never walk into a provable 1-ply loss at play time (guard),
   - never over-value a proven-lost / proven-won leaf during search (override).
   What it gives up vs full TSS: the search won't *force-explore* the covering
   move, so it relies on the network's policy to already surface it among the
   nucleus. Given dense_cnn already feeds hot-cells planes to the net, the
   covering move is usually in-policy — so the marginal value of injection over
   override+guard is smaller for dense_cnn than it was for hexgt.

2. **Shared threat module as the deliverable.** Even if only §3.1 lands,
   factoring `threats.rs` into a shared module is independently worthwhile: it
   removes the latent drift risk of two lineages re-deriving threat semantics,
   and makes 3.2–3.4 incremental.

3. **Python-side prior reweighting (cheapest, weakest).** Re-rank/boost tactical
   priors inside `inference.py:evaluate_model1_payload` before Rust parses them
   — no Rust rebuild. This is a soft nudge, not a guarantee (PUCT can still
   ignore it, and it can't force a leaf verdict). Useful only as a quick
   experiment, not a substitute for the override/guard.

4. **Full parity.** §3.1–§3.4 (+ optionally §3.5 later). ~1 week. Only worth it
   if measurement shows override+guard leaves material tactical losses on the
   table.

---

## 5. Verdict

**Overall difficulty: Moderate.** Effort estimate for an engineer fluent in this
Rust MCTS:

- **Lighter path (override + guard + shared module): ~1–2 days.**
- **Full parity (+ expansion injection, crop handling, tests): ~1 week.**
- Optional threat-plane enrichment: +~1–2 days **and a retrain** (separate
  decision).

**Why it's cheaper than a naive estimate:** the threat core is already shared
engine code, already linked into dense_cnn's binary, and provably decoupled from
the graph representation. The work is *adapter wiring into one more Rust MCTS*,
not a reimplementation.

**Main risks:**
1. Editing the hottest tree path (`select_or_materialize_edge`,
   prior-staging) for injection — contained to §3.4, mitigated by keeping the
   no-threat path untouched and gating injection on `windows().has_threats()`.
2. The crop limit on injection — real but graceful, and consistent with
   dense_cnn's existing action space.
3. Operational: must not rebuild `_rust` against the live run's worktree.

**Is it worth it for a fixed-board CNN?** Partly. The reframing question is "does
dense_cnn need *infinite-board* threat handling?" — and the answer is **no**: its
crop already bounds where it can act, so the exotic out-of-crop cases TSS handles
on hexgt's infinite graph are moot for dense_cnn. But the *core* tactical wins —
not playing into a forced loss, not mis-valuing a proven leaf — are exactly
dense_cnn's known weakness and are delivered by the **crop-independent, no-retrain**
override + guard at low cost. That subset is clearly worth it. Expansion
injection is a smaller marginal gain for dense_cnn than it was for hexgt and
should be gated on measured benefit. Threat-plane enrichment is a model-quality
lever that competes with simply training longer and should be evaluated on its
own, not bundled into the TSS port.

**Recommendation:** do §3.1 + §3.2 + §3.3 first, measure tactical blunder rate
against the current `dense_cnn_rl_main1` checkpoint, and only pursue §3.4 if the
residual justifies touching the hot path.

---

## Appendix: key file references

**Shared / engine**
- `packages/hexo_engine/rust/src/tactics.rs:342-486` — `WindowStore`
- `packages/hexo_engine/rust/src/tactics.rs:189-203` — threat predicate (`count >= 4`)

**hexgt TSS (port sources)**
- `packages/hexo_models/hexgt/rust/src/threats.rs:54-199` — verdict / hitting-set / analyze / tactical_cells
- `packages/hexo_models/hexgt/rust/src/mcts_tree.rs:833-873` — `split_tactical`
- `packages/hexo_models/hexgt/rust/src/mcts_tree.rs:513-520` — forced-edge first visit
- `packages/hexo_models/hexgt/rust/src/mcts.rs:623-633` — leaf override hook
- `packages/hexo_models/hexgt/rust/src/mcts.rs:913-992` — move-selection guard
- `packages/hexo_models/hexgt/rust/src/features.rs:131-193` — v3 threat features (graph-coupled; NOT in scope)

**dense_cnn (port targets)**
- `packages/hexo_models/rust/src/lib.rs:4-17` — single native module for all three lineages
- `packages/hexo_models/dense_cnn/rust/src/encoding.rs:196-210` — existing `windows()` use (hot-cells planes); `:91, 261-296` crop projection
- `packages/hexo_models/dense_cnn/rust/src/mcts_tree.rs:125-127, 185, 400, 509-545` — nucleus widening, candidate staging, materialization gate
- `packages/hexo_models/dense_cnn/rust/src/mcts.rs:295-330, 522-528, 599-614` — action selection, inline leaf backup, raw-vs-export policy split
- `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/inference.py:267-335` — evaluator callback (Python reweighting option)

**Design history**
- `docs/analysis/HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md` — authoritative TSS design (§1.4 the `Owned`-prior caveat)
- `docs/analysis/HEXGT_TSS_IMPL_REVIEW.md`, `docs/analysis/HEXGT_TSS_VERIFICATION.md` — review + engine verification
- `notes.md` — live Model 3 run (TSS always-on)
