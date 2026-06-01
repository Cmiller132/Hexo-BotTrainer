# Hexformer Rewrite Plan — Dynamic GNN + Transformer Hybrid (Model 2)

**Status:** Design / planning only. No implementation, no pipeline changes. The
dense_cnn 96×8 run is live on the GPU — this plan must not compete for GPU and
nothing here is to be run during that run.

**Date:** 2026-06-01 (revised: dynamic policy, active-window tactics, dynamic
candidate set, no padding/no-TRT)
**Author:** read-only analysis pass
**Reference to emulate (discipline, not architecture):** `hexo_models.dense_cnn`
(Model 1), 96ch×8blk, currently beating SealBot best-50ms (~92% @ e17, `MEMORY.md`).

---

## 0. TL;DR / Executive summary

- **Delete `hexformer_ar` and build Model 2 from scratch**, modeled on
  `dense_cnn`'s *discipline* (package structure, config system, `ModelPlugin`
  pattern, replay/checkpoint hygiene, test discipline, Rust↔Python MCTS handoff).
  `hexformer_ar` is untested, bloated, off-pattern, and is **not** a foundation
  (§1). We do not carry its model, scaffolding, trainer, samples, or Rust crate
  forward.
- **The model is a *truly dynamic* typed heterogeneous GNN → transformer
  hybrid.** Nodes: **all** placed stones, the dynamic candidate set of empty
  cells, the active-window tactical tokens, and side/goal tokens. No padding, no
  fixed shapes, no `max_candidates`, no fallback logits, and **no cap/top-k on
  the model's candidate set**. Variable node/edge counts are first-class (§3).
- **Two layers — model scores ALL candidates; MCTS may nucleus-widen within
  that support.** (1) The policy produces logits over the **full** candidate set
  (no cap). (2) MCTS may *expand/visit* only a top-p (nucleus, mass-based) subset
  *inside* that support — mirroring dense_cnn's `widening_policy_mass=0.95`. This
  is search-side and is **not** a contradiction of "no candidate cap": the model
  still scores every candidate and the training target is still defined over the
  full support (un-expanded children get ~0 visits, dense_cnn discipline). See
  §6.6.
- **The policy is dynamic and per-legal-move (pointer/CSR), not a 1681 dense
  crop.** The head emits one logit per candidate node; this maps *directly* onto
  the Rust MCTS priors contract, which is already per-legal-move CSR
  (`legal_row_offsets` + `priors_bytes`, §2.6).
- **The candidate set is a union of an engine-grounded active-window component
  and a small local-neighborhood component:**

  > **`candidate_set(position) = { empty cells in ANY active window of either
  > player }  ∪  { empty cells within hex-distance ≤ n of ANY placed stone }`**

  with **`n` a single config parameter, default `n = 2`, tunable in `[2, 8]`**
  (same hex-distance metric as the engine's legality radius, so `n = 8` ≈ the
  engine's full legal set). The active-window component guarantees every far
  threat-completion / must-block cell (up to 5 line-cells from a stone) and all
  developing-line cells; the n-radius component adds the local "start a new
  window nearby" development moves the active-window rule alone cannot reach. An
  *active window* is the engine's concept — a length-6 line window containing
  stones of exactly one player (`tactics.rs:183-186`). See §4.
- **Tactical-window tokens** are the **count-3/4/5 active windows of both
  colors** (developing threes + live threats + win/block windows). No forks, no
  other window types. The candidate-cell filter uses *all* active windows
  (count ≥ 1); the window *tokens* are the higher-count (3/4/5) subset (§5).
- **No TensorRT.** A truly dynamic GNN does not export to TRT. Inference is
  **torch (FP16 ok)**; the throughput phase is *"make the dynamic GNN fast
  enough in torch,"* not *"pad/export it."* The attention-bias / fixed-shape
  formulation is explicitly a **non-GNN approximation and is rejected** for this
  design (§4.3, §6.1).
- **"Drop-in" means PIPELINE compatibility, not matching dense_cnn's tensor
  shapes.** Model 2 slots into the existing training / MCTS / replay / eval
  pipeline (plugin protocol, Rust↔Python evaluator callback, checkpoint payload,
  SealBot eval harness, replay window/shuffle discipline). Its policy shape is
  deliberately different (dynamic); its value head stays 65-bin because that's
  pipeline-convenient, not because shapes must match (§2).
- **Major sample-gen + training rewrite** (say so): emulate dense_cnn's
  raw-fact NPZ + replay discipline, but the Rust graph/active-window sample-gen,
  the expand-to-graph step, variable-size graph collation, and the trainer are
  largely new (§7). Behavioral cloning from dense_cnn shards is useful but is a
  **conversion rewrite**, not free reuse (§8).

---

## 1. The current `hexformer_ar` and why it is deleted

Package `packages/hexo_models/hexformer_ar/` (~3,200 Py + ~3,500 Rust lines). It
is a sparse hybrid (local hex-CNN → GraphGPS token stack → candidate-pointer
policy) — conceptually adjacent to the target, but:

- **Never trained / wholly unvalidated.** No `runs/hexformer_ar` artifacts;
  `MEMORY.md` records dense_cnn only. The unused `HexformerOutputs` dataclass
  ([architecture.py:15-23](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:15)) and vestigial "AR" name signal abandoned
  drift.
- **Off-pattern / bloated.** Its GNN is a per-batch-element Python `for` loop
  (`_edge_aggregate`, [architecture.py:134-164](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:134)); replay is in-memory
  zlib+JSON ([samples.py:42-53](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/samples.py:42)); the checkpoint persists `sample_buffer`
  — the opposite of dense_cnn discipline.
- **Diverged contracts.** 3-class WDL value, fixed-shape padded candidates, a
  different aux-head set.

**Verdict: DELETE.** Build Model 2 fresh, modeled on dense_cnn's patterns.
"Delete" concretely (part of the build work, not this planning pass): remove
`packages/hexo_models/hexformer_ar/`, its entry point + source-include lists in
`packages/hexo_models/pyproject.toml`, its `#[path]` include +
`sys.modules` registration in `packages/hexo_models/rust/src/lib.rs:7-9,24-29`,
`configs/hexformer_ar.toml`, and any `hexformer_ar` tests. We may *consult* (not
import) its hex-D6 group and coordinate-packing math, re-deriving and re-testing
them in the new package.

---

## 2. Pipeline compatibility — what Model 2 must slot into

dense_cnn is the **template for discipline** and the source of the **pipeline
interfaces** Model 2 must satisfy. The requirement is **drop-in PIPELINE
compatibility**, not matching dense_cnn's exact tensor shapes.

### 2.1 What must match (pipeline interfaces)

- **Plugin / registry** (`hexo_train/registry.py:24-103`): implement the
  `ModelPlugin` Protocol — `name`, `build_model(game_spec, config)`,
  `training_component_overrides(*, defaults, config, shared, model) ->
  ComponentOverrides`; optional `calibrate_performance`, `generate_selfplay`,
  `evaluate_epoch`. Register under the `"hexo_train.models"` entry-point group.
- **Rust↔Python MCTS evaluator callback** (the load-bearing interface,
  `dense_cnn/rust/src/mcts_eval.rs:315-390`): Rust calls
  `evaluator.call1((payload,))`; Python returns `{"values_bytes": <N f32, clamped
  [-1,1]>, "priors_bytes": <f32, one per legal/candidate move, row-major in
  legal_row_offsets order>}`. Rust validates finite/nonneg/positive-mass and
  normalizes. **This contract is already per-move CSR** — a dynamic per-candidate
  policy maps onto it directly (§3.4). Model 2 ships its *own* payload contents
  (a graph, not planes) but reuses the **buffer-protocol zero-copy transport**
  pattern (the `PlaneBuffer` `#[pyclass]`, `mcts_eval.rs:48-100`) for its node/
  edge tensors.
- **Checkpoint payload** (`dense_cnn/checkpoints.py:80-105`): `{"model",
  "model_state","optimizer_state","train_state","epoch","metadata"}`; reject a
  non-None `sample_buffer`, incompatible `model_state`, and missing `.txt`
  pointer; support `.txt` indirection.
- **Replay discipline** (`dense_cnn/{replay.py,compact_io.py}`): raw-fact
  columnar NPZ shards + `int64` offset arrays + a `SCHEMA_VERSION`; power-law
  replay window (`replay.py:523-535`); md5 train/val split; batch-aligned output
  shards; per-shard JSON sidecars; **per-epoch D6 applied at read/expand time**.
- **Config** (`dense_cnn/config.py`): frozen dataclass sections; per-section
  unknown-key rejection; per-scalar coercion; no range validation.
- **Eval harness** (`dense_cnn/{evaluation.py,player.py}`): `evaluate_epoch` →
  `hexo_runner.modes.match.run_match` vs `SealBotPlayer`; `use_trt=False` for
  eval; opening-diversity controls.

### 2.2 What deliberately differs from dense_cnn

- **Policy:** dynamic per-candidate logits (variable length), **not** `(N,1681)`.
  Remove all 1681 dense-policy framing for this model (§3.4).
- **Inputs:** a typed heterogeneous **graph** (variable nodes/edges), not a
  `(13,41,41)` plane stack. The 41×41 crop, `BOARD_AREA=1681`, and `coord_to_flat`
  mapping are dense_cnn-only and do not apply to Model 2.
- **Value:** kept as the **65-bin distributional head** over `linspace(-1,1,65)`
  (`dense_cnn/losses.py:20-30`) — not because shapes must match, but because it's
  pipeline-convenient and proven; reuse the binning math verbatim.

### 2.3 Parameter budget for a fair comparison

Size to the running 96×8 baseline (~2.1M params by hand estimate; the "~2.6M" in
prior briefs is not in code). Verify with `sum(p.numel())` and land within ~10%.

---

## 3. The engine rules (grounded) — what the model and tactics are built on

All quotes from `packages/hexo_engine/rust/src/`.

- **Win = six-in-a-line.** `WINDOW_LEN = 6` ([tactics.rs:14](packages/hexo_engine/rust/src/tactics.rs:14)); a window is
  a length-6 segment along one of three axes Q `(1,0)`, R `(0,1)`, QR `(1,-1)`
  ([tactics.rs:23-53](packages/hexo_engine/rust/src/tactics.rs:23)). A win is an active window fully filled by one
  player: `is_win_for` = `active_player()==Some(p) && count(p)==6`
  ([tactics.rs:206-208](packages/hexo_engine/rust/src/tactics.rs:206)). "A win is checked after every single placement."
  ([state.rs:9](packages/hexo_engine/rust/src/state.rs:9)). Each placement touches exactly 18 windows (3 axes × 6
  offsets, [tactics.rs:14-17](packages/hexo_engine/rust/src/tactics.rs:14)).
- **ACTIVE WINDOW (the core concept).**
  > `is_active` — *"True when the window contains stones from exactly one
  > player."* ([tactics.rs:183-186](packages/hexo_engine/rust/src/tactics.rs:183))
  >
  > `active_player` returns `Some(p)` only for `(true,false)`/`(false,true)`
  > occupancy of the two player masks ([tactics.rs:172-181](packages/hexo_engine/rust/src/tactics.rs:172)).

  So an active window has ≥1 stone of exactly one player and **zero** of the
  other; the remaining cells are empty and the window is still *completable*. A
  single opponent stone makes it inactive (dead) — test
  `blocked_windows_are_not_threats` ([tactics.rs:620-636](packages/hexo_engine/rust/src/tactics.rs:620)).
- **Threat = one-sided count ≥ 4.** `threat_player` = active player with
  `count(player) >= 4`; `is_threat` ([tactics.rs:189-198](packages/hexo_engine/rust/src/tactics.rs:189)).
- **Count is a popcount of the 6-bit mask, NOT contiguous.** `count(player) =
  mask(player).count_ones()` ([tactics.rs:133-136](packages/hexo_engine/rust/src/tactics.rs:133)). **Implication:** gapped /
  broken shapes like `XX_XX` inside a 6-window are already counted as a count-4
  threat — there is no separate "broken threat" case to detect. (This corrects
  earlier brainstorming.)
- **`WindowStore` maintains active/threat/win windows incrementally** and can
  enumerate them cheaply: `entries()` ([tactics.rs:378](packages/hexo_engine/rust/src/tactics.rs:378)),
  `threat_entries(player)` ([tactics.rs:386-389](packages/hexo_engine/rust/src/tactics.rs:386)), `threats()`
  ([tactics.rs:392-395](packages/hexo_engine/rust/src/tactics.rs:392)); per-window `empty_cells()`
  ([tactics.rs:154-156](packages/hexo_engine/rust/src/tactics.rs:154)), `count`, `active_player`, `stone_cells`. The store only
  holds *touched* windows (those that ever received a stone,
  [tactics.rs:343-345,419](packages/hexo_engine/rust/src/tactics.rs:343)), so active windows ⊆ touched windows and
  enumeration is O(#touched).
- **Legal moves are already locality-bounded.** Non-opening legal moves = empty
  cells within `LEGAL_RADIUS = 8` of any stone ([legal.rs:10-11,124-128](packages/hexo_engine/rust/src/legal.rs:10);
  recompute reference `state.rs:559-569`). One stone ⇒ 216 legal cells
  ([legal.rs:224](packages/hexo_engine/rust/src/legal.rs:224)).
- **Two-stone autoregressive turns.** `Opening` (Player 0 forced at `(0,0)`) →
  `FirstStone` → `SecondStone` (same player places the 2nd) → control passes
  ([state.rs:46-56,312-330](packages/hexo_engine/rust/src/state.rs:46)). If the first stone wins, the second is never
  played ([state.rs:9-10,304-310](packages/hexo_engine/rust/src/state.rs:9)).
- **No captures, no draws, unbounded board.** Stones are only placed/undone (no
  removal); `GameOutcome` has a winner only, "Hexo has no normal draw"
  ([state.rs:64-71](packages/hexo_engine/rust/src/state.rs:64)); the board is a "Sparse unlimited board"
  ([state.rs:98](packages/hexo_engine/rust/src/state.rs:98)) on an "unlimited hex grid" ([coord.rs:1-9](packages/hexo_engine/rust/src/coord.rs:1)), bounded
  only by the i16 coordinate range. **There are no playing-field edges** in
  normal play — the "41×41 board" is only dense_cnn's crop, irrelevant to a
  relative-coordinate graph model. (No edge/boundary window type is needed.)
- `hex_distance` = cube max-norm `max(|dq|,|dr|,|ds|)` ([coord.rs:77-82](packages/hexo_engine/rust/src/coord.rs:77)).

---

## 4. The candidate set — the move vocabulary at each node

### 4.1 Definition (active-window union local-neighborhood)

```
candidate_set(position) = { empty cells in ANY active window of either player }   (A)
                         ∪ { empty cells within hex-distance ≤ n of ANY stone }   (B)
```

- **`n` is a single clean config parameter, default `n = 2`, tunable in `[2, 8]`.**
  It uses the **same hex-distance metric as the engine's legality radius**
  (`hex_distance`, cube max-norm, `coord.rs:77-82`; `LEGAL_RADIUS = 8`,
  `legal.rs:10-11`), so `n` is directly comparable: `n = 8` ≈ the engine's full
  legal set, `n = 2` is a tight local neighborhood.
- **Component (A)** — Rust: iterate `board().windows().entries()`, keep
  `is_active()` (either color, any count ≥ 1), collect `empty_cells()`, dedupe.
- **Component (B)** — Rust: union `coords_within_radius(stone, n)` over all
  stones (`coord.rs:87-96`), keep empties. This is the same construction the
  engine uses for legality at radius 8, just with a tunable, smaller `n`.

The two components have **distinct, complementary roles** (§4.2). The set is the
union; **no top-k, no `max_candidates`, no fallback logits on the candidate set
itself.** Count is dynamic. (Distinct from this: MCTS may later nucleus-widen
*which* of these candidates it expands — a search-side subset *inside* the full
support, not a cap on the model's moveset. See §6.6.)

### 4.2 Why the union — each component's role

- **(A) active windows → guarantees all far threat / block / developing cells.**
  Because win length is 6, a window-completing (threat-completion or must-block)
  cell can sit up to **5 line-cells from the nearest stone** of that window —
  *beyond* a small `n`-radius. Component (A) includes every such cell by
  construction (it belongs to an active window), at any distance, so the policy/
  search can always answer or complete a threat. (A) also includes every
  developing-line cell of an existing one-sided line.
- **(B) n-radius → adds local "create a NEW window" development moves.** This is
  the failure mode of an active-window-**only** set: (A) can only *extend or
  block existing* one-sided lines; it **cannot start a new window in a fresh
  direction.** **Concrete move-2 example:** with a lone stone at origin, the only
  active windows are the length-6 windows along the 3 axes through origin, so an
  active-window-only candidate set is limited to cells **on those 3 axis lines**
  — far too restrictive (it can never play off-axis to begin a new line). The
  `n = 2` neighborhood adds the nearby empties needed to start new windows.
- **Why `n = 2` as default.** `n = 2` is enough to respond to any threat
  (component (A) already covers completions; (B) just needs local development),
  `n ≈ 3` gives more developmental reach, and full `n = 8` (engine legal) is more
  than needed — so we **test `n = 2`** but keep it trivially tunable to any value
  in `[2, 8]` for later sweeps.

**Why prune at all (model tractability, not legality).** dense_cnn's conv policy
head emits all 1681 crop logits "for free" from a fixed feature map; a token
GNN/transformer instead pays ~O(N²) attention + per-node message cost in the
**number of candidate tokens** N. The union bounds N to the connection-relevant
cells (existing-line cells + local development) — the move vocabulary a 6-in-a-
row game actually uses. Because the dynamic policy is defined over these tokens,
**the candidate set is also exactly what MCTS expands at that node** (priors are
emitted per candidate; the Rust search children = candidate tokens). `n` is the
single knob trading tractability against breadth; the §4.6 coverage gate decides
whether `n = 2` is wide enough.

### 4.3 Rejected alternatives

> The **active-window-only** rule (no `n`) is **rejected** as too restrictive —
> it cannot start a new window in a fresh direction (the move-2 example above).
> The **attention-bias / fixed-shape padded** formulation is **rejected** as a
> non-GNN approximation: it cannot represent the truly dynamic, unbounded graph
> structure the user wants and would reintroduce caps/padding. Both are recorded
> here only to note they were evaluated and declined.

### 4.4 Opening special case

On an **empty board there are no stones, hence neither component (A) nor (B)
yields anything → an empty candidate set.** Handling:

- **Move 1** is the engine-forced `Opening` placement at `(0,0)`
  ([rules.rs:16-23](packages/hexo_engine/rust/src/rules.rs:16), [state.rs:224-228](packages/hexo_engine/rust/src/state.rs:224)) — a single candidate; no model choice
  needed. The pipeline can hard-code it (matching engine legality).
- **Move 2 onward:** once the center stone exists, the set is non-empty from both
  components — (A) the empty cells of the 18 active length-6 windows through
  `(0,0)`, **and** (B) the empty cells within hex-distance `n` of `(0,0)` (the
  off-axis local development the active-window component alone would miss, §4.2).
  No special code needed beyond move 1.
- **Safety net:** if any non-terminal position ever yields an empty candidate set
  (should only be the pre-opening empty board), fall back to the engine's legal
  move list for that node and log it loudly (no silent divergence between
  candidates and legality).

### 4.5 Rust parity (training ↔ play)

Both components — the active-window enumeration (A) **and** the `n`-radius
neighborhood (B) — must run in **one shared Rust path** (parameterized by the
single `n`) used by **both** sample-gen and live MCTS, so the move vocabulary at
a node is identical in training data and in play. (Mismatch would train the
policy over a different support than search expands.) This is a single function
over the engine `WindowStore` + `coords_within_radius`, called from both
`sample_gen` and the MCTS encoder, with `n` threaded from config to both.

### 4.6 Validation gates (before locking the design)

Replay dense_cnn's recorded games (`runs/.../selfplay/*.hxr` / shards,
read-only) and measure:

1. **Completeness/safety at `n = 2`:** fraction of dense_cnn's **actually-played**
   moves and **MCTS-visited** moves that fall inside `candidate_set` with the
   default `n = 2`. **Target ≈ 100%.** (dense_cnn searches the full radius-8 legal
   set, so this directly tests whether the union ever excludes a move a strong
   player used.) **Sweep `n = 2..8`** in this same check and report the coverage
   curve — this empirically sets the smallest `n` that achieves ~100% coverage
   and flags whether the default needs raising. A genuine miss at `n = 8` (the
   full legal radius) would be a design red flag.
2. **Cost distribution at `n = 2`:** the **size distribution** of `candidate_set`
   across game phases (opening / midgame / endgame): report median, typical, and
   p95 candidate counts, plus active-window-token counts. This sets the GNN's
   per-node cost and the realistic batch sizes for the throughput phase. (Re-run
   at the chosen `n` if the sweep moves it off 2.)
3. **Node- AND edge-count distribution at `n = 2`:** for each position, count
   total nodes (stones + candidates + active-window tokens + side/goal) and
   **total edges by type** under the bounded construction (§6.3); report median /
   typical / p95 per game phase. **This is the explicit no-explosion gate** — it
   empirically confirms the edge count stays **linear in (#nodes +
   #active-windows)** (no same-axis clique) **before** committing to the GNN cost.
   A super-linear edge growth here is a hard stop to revisit edge construction.

All three are **gates**: (1) proves the prune is complete/safe; (2) proves the
candidate (node) cost is bounded; (3) proves the **edge** cost does not explode.
Together they prove the dynamic graph is trainable and searchable at acceptable
throughput before the GNN is built.

---

## 5. Tactical-window tokens (active windows by count)

The tactical-window **tokens** are the **count-3/4/5 active windows of BOTH
colors** — the connection-relevant subset of the same active windows used for
the candidate filter. Typed by `(owner ∈ {current, opponent}, count ∈ {3,4,5})`:

| token | owner | count | engine meaning | detection (from `WindowStore`) |
|---|---|---|---|---|
| **T0 win-in-1** | current | 5 | single empty cell completes the six → immediate win | active window, `count(current)==5`, `empty_mask.count_ones()==1` |
| **T1 must-block** | opponent | 5 | opponent's single empty cell wins next → must block | active window, `count(opp)==5`, single empty |
| **T2 live threat** | current | 4 | one-sided count ≥4 → live threat (per `threat_player`) | `threat_entries(current)` filtered to `count==4` |
| **T3 live threat** | opponent | 4 | opponent live threat to answer | `threat_entries(opponent)` filtered to `count==4` |
| **T4 developing three** | current | 3 | developing line | active window, `count(current)==3` |
| **T5 developing three** | opponent | 3 | opponent developing line | active window, `count(opp)==3` |

Relationship to the candidate set: the candidate set's active-window component
(§4 (A)) uses **ALL active windows (count ≥ 1)** for their empty cells; the
**window tokens are the higher-count (3/4/5) subset**. Count-1/2 active windows
still contribute their empty cells as candidates (via (A), and also via the
`n`-radius component (B)) but are not emitted as tactical tokens (they carry
little structure beyond locality, which the candidate node's own features and the
`n`-radius edges already express).

**No forks / double-threats are computed.** A cell sitting in multiple active
windows *is* a fork; the GNN/transformer learns this implicitly because that
candidate node attends to (is edged to) multiple window tokens (§6.3). **No
other window types** (broken/gapped is already covered by popcount §3; no
intersection/VCF/dead-line/edge tokens). **No top-k on the tokens** — all
count-3/4/5 active windows are emitted; the count is dynamic. (MCTS nucleus
widening, §6.6, is a separate search-side concern and does not drop window
tokens.)

Token features per active-window node: owner (current/opponent), count
(3/4/5 one-hot), axis (Q/R/QR one-hot), the window's empty-cell count, and a
relative anchor coordinate (e.g. the window's start or centroid in
center-relative axial coords). Detection cost is O(#touched windows) and largely
already paid by the engine's incremental threat tracking.

---

## 6. Model 2 — the dynamic GNN + transformer hybrid

Working name **`hexgt`** (Hex Graph-Transformer). It is fundamentally a **typed
heterogeneous GNN**; a transformer adds global attention; heads produce a dynamic
policy + the 65-bin value (+ aux). **Truly dynamic: no padding, no fixed shapes,
no caps.**

### 6.1 Why dynamic, and why no TRT

- Node and edge counts vary per position (stones grow; candidate/window counts
  vary, §4.6). The user requires handling **truly unbounded play**, so we do not
  pad to a max or cap candidates.
- A dynamic message-passing GNN uses `scatter`/segment reductions over
  variable-length edge lists, which **do not export cleanly to TensorRT**.
  Therefore **inference is torch (FP16 acceptable), no TRT.** The throughput
  phase (§9, Phase 5) is *"make the dynamic GNN fast enough in torch,"* not
  *"pad/export it."* Concretely: batch positions by **packing into one big
  disjoint graph** (PyG-style: concatenate all nodes across the batch with a
  per-node `graph_id`; message passing via segment scatter; attention masked
  block-diagonally per graph). This runs efficiently on GPU in eager torch; the
  costs are kernel-launch overhead and the absence of TRT fusion — accepted and
  measured.
- The attention-bias / fixed-shape realization is **rejected** (§4.3): it is a
  non-GNN approximation that cannot represent the dynamic structure.

### 6.2 Node types and features

| type | source | count | features |
|---|---|---|---|
| **side/goal** (1–2) | `current_player`, `phase`, global counts | fixed small | side-to-move, phase one-hot, stone counts, move number |
| **stone** | ALL placed stones (no budget) | = #stones | owner (own/opp), recency (`hist_idx`), relative axial coord |
| **candidate** | §4 candidate set | dynamic | relative axial coord, #active windows through this cell (by owner/count), is-it-a-window-completing cell |
| **active-window** | §5 (count 3/4/5, both colors) | dynamic | owner, count one-hot, axis one-hot, empty-cell count, anchor coord |

All coordinates are **center-relative axial/cube** (re-derived, re-tested helpers;
the engine's `pack_coord`/`hex_distance` are the references, `legal.rs:24-35`,
`coord.rs:77-82`). The unbounded board means relative coords never hit an edge.

### 6.3 Typed edges — bounded construction (NO same-axis cliques)

**Make-or-break risk:** naïve "same-axis-line membership" edges that connect
**every pair** of stones/candidates sharing a line form an **O(N²) clique** that
explodes on dense boards. This is the specific stone↔stone-line failure the user
flagged. The design **forbids all-pairs co-linearity edges.** Instead:

**Route all line / co-linearity relationships THROUGH WINDOW NODES as the hub.**
Each length-6 window has ≤6 cells, so membership edges are **O(#windows × 6)** —
bounded, not quadratic. A stone and a candidate that share a line are then **2
hops apart via the shared window node**, with no direct clique. The window node
is the line's representative; co-linearity is learned through it.

Edge types, each with an **explicit cardinality bound**:

| edge type | endpoints | bound | role |
|---|---|---|---|
| **hex-adjacency** | node ↔ node within hex-distance 1 | **≤ 6 per node** (hex has 6 neighbors) → O(#nodes) | local locality |
| **stone↔window membership** | window ↔ its one-color stones | **≤ 6 per window** → O(#windows·6) | line/group structure via hub |
| **candidate↔window membership** | window ↔ its empty cells | **≤ 6 per window** → O(#windows·6) | a candidate "sees" every threat/extension it's in (forks learned implicitly) |
| **recency** | stone ↔ immediately-preceding/following stone in placement order | **chain, 2 per stone** → O(#stones) | temporal structure |
| **side/state/goal context** | side/goal node ↔ all nodes | O(#nodes), 1 hub | global broadcast |

**If a DIRECT on-line edge between cells is ever wanted** (it is not required for
the MVP — the window hub suffices), restrict it to a **nearest-neighbor-along-
line chain**: each node links only to its immediate predecessor/successor on that
axis line — **O(N) per line, never a clique.**

**Total-edge bound:** `O(#nodes·6) + O(#active_windows·6) + O(#stones) +
O(#nodes)` = **linear in (#nodes + #active_windows)** — no quadratic term. With
the side/goal hub the only all-to-one edges. (#active_windows here = the windows
materialized for membership; the touched-window count, O(#stones) in practice.)

Message passing is type-conditioned: `m_{j→i} = φ_{edge_type}(h_j, h_i,
rel_coord_{ij})`, aggregated per target node (segment scatter) over `L_gnn`
rounds. Then a transformer block set provides global attention (context
self-attention over {side, stone, window}; candidate→context cross-attention;
optionally one candidate↔candidate sparse self-attention), all masked
block-diagonally per graph in a packed batch. **Note:** the transformer's
candidate↔context attention is itself ~O(#candidates·#context) per graph — the
nucleus widening in §6.6 bounds *search* cost but not this dense attention, so
the Phase-1 node/edge-count gate (§4.6) is what guards the GNN/attention cost.

### 6.4 Heads (dynamic policy + pipeline-compatible value/aux)

| head | output | notes |
|---|---|---|
| **policy** | one logit per **candidate** node | dynamic length; softmax over the candidate set → emitted as `priors_bytes` in `legal_row_offsets` order (§2.6). No 1681 dense form. |
| **value** | `(N,65)` 65-bin distributional | reuse `linspace(-1,1,65)` + `binned_value_loss` from dense_cnn |
| **opp_policy** (aux) | one logit per candidate | opponent's policy target, same dynamic shape |
| **stvalue_<h>** (aux, optional) | `(N,65)` per horizon | reuse dense_cnn binning + mask |

Training policy target: dense_cnn-style visit counts, but **defined over the
candidate nodes** (not a 1681 grid). Because MCTS expands exactly the candidate
set (§4.2), every visited move is a candidate node — the target is well-defined
with no scatter to a crop. Drop hexformer's WDL/distance/threat/lookahead heads.

### 6.5 D6 symmetry (cleaner on a dynamic graph)

- Re-derive/re-test the hex-D6 group (12 elements). Apply **at read/expand time**
  via a per-(run,epoch) symmetry vector (dense_cnn discipline).
- D6 acts on each node's **relative coordinate** and permutes/reflects the
  **window axis labels** (Q/R/QR map among themselves under rotation/reflection).
  Node and edge **identity is permutation-invariant**, so the graph is unchanged
  except for rotated coords and relabeled axes.
- Because the rep is relative-coordinate and **not bound to a square crop**, the
  dense_cnn corner-spill problem disappears: the **full D6 group is usable with
  no identity fallback**. This is a real advantage of the dynamic graph rep.
- **Equivariance test (non-negotiable):** applying D6 to the input graph and
  inverse-D6 to the policy output must equal the un-augmented forward (within fp
  tolerance) for all 12 elements — the test that prevents subtly poisoning the
  model (the dense_cnn D6 lesson, `MEMORY.md`).

### 6.6 Two layers: full-support scoring vs MCTS nucleus widening

These are **distinct concerns** and must not be conflated:

- **Layer 1 — model support (no cap).** The policy head emits a logit for
  **every** candidate in the full set (§4 union). The softmax and the training
  policy target are defined over this **full support**. No top-k, no cap, no
  fallback — the model always scores its entire moveset. This is what the
  earlier "no top-k / no candidate cap" statements mean.
- **Layer 2 — MCTS expansion (top-p / nucleus, mass-based).** The Rust search
  may *expand/visit* only a subset of that support, selected by **nucleus
  (top-p) widening**: materialize children in descending prior until their
  cumulative prior mass reaches a threshold (mirroring dense_cnn's
  `widening_policy_mass = 0.95`). **Prefer nucleus/top-p (mass-based) over top-k
  (count-based)** so the breadth adapts to how peaked the policy is. This
  operates **strictly inside Layer 1's full support** — it never adds an
  out-of-support move and never changes the model's scores.

**Why this is not a contradiction of "no candidate cap":** the model still
scores all candidates; the policy *target* is still over the full support;
un-expanded children simply receive ~0 visits — exactly dense_cnn's discipline
(the visit-count target naturally concentrates mass on expanded children while
leaving the rest near zero). Widening bounds *search* branching/compute, not the
model's moveset or its training signal. (The dense GNN/attention cost over the
full candidate set is bounded separately by the §4.6 node/edge-count gate, not
by widening.)

The widening threshold is a search/eval config knob (default ~0.95, like
dense_cnn), tuned in Phase 5/7; it must be identical between self-play and the
head-to-head eval so the comparison is fair (§10).

---

## 7. Sample-gen + training rewrite (scope, honestly)

**Emulate dense_cnn's discipline** (raw-fact columnar NPZ + offsets +
`SCHEMA_VERSION`, power-law replay window, md5 split, batch-aligned shards, JSON
sidecars, per-epoch D6-at-read, checkpoint hygiene, config/plugin/test patterns).

**New, largely-rewritten work:**
1. **Shared Rust candidate/active-window path** (§4.5): one function over the
   engine `WindowStore` producing the candidate set + the count-3/4/5 window
   tokens, called by both sample-gen and live MCTS.
2. **Rust sample-gen (representation-agnostic for MVP):** shards stay raw-fact
   only — the existing compact schema (`stones_qr`, `legal_ids`, history, plus
   windows recomputable from stones) is sufficient. **DECIDED:**
   **recompute-at-expand for the MVP** — the candidate set, active-window tokens,
   typed nodes, and typed edges are **rebuilt at expand time** from the raw-fact
   shards (via the shared Rust candidate/active-window path, item 1). **No new
   `SCHEMA_VERSION` for the MVP.** Caching the graph/edges into a new
   `SCHEMA_VERSION` is a **later optimization, gated on the model being proven to
   work** (§9 Phase 8), not MVP. **Accepted MVP tradeoff:** expand-time recompute
   spends CPU every epoch (the graph is rebuilt per read), but keeps the shard
   format representation-agnostic while the graph representation is still in flux
   — flexibility over CPU now, caching later once the rep is locked.
3. **Expand step (`expand_row_to_graph`, new — the MVP graph constructor):**
   compact raw-fact row + D6 symmetry → (recompute candidate set + active-window
   tokens + typed nodes/edges) → typed node tensors + typed edge lists + dynamic
   policy/opp targets + 65-bin value/stvalue targets. Replaces dense_cnn's
   `expand_*_to_planes`. This is the per-epoch recompute path; it must use the
   same shared Rust candidate/active-window function (item 1) as live MCTS so
   training support == search expansion (§4.5).
4. **Variable-size graph collation/batching (new):** pack a batch into one
   disjoint graph with `graph_id`; deterministic, byte-stable, tested.
5. **New trainer (`HexgtTrainer`):** consumes packed graph batches; **same loss
   weights, AMP, grad-clip, optimizer (AdamW), and reporting discipline as
   dense_cnn**, with a dynamic (variable-length, per-graph-normalized) policy CE.
6. **New inference module:** graph payload → per-candidate `priors_bytes` +
   `values_bytes` (§2.6), using the zero-copy buffer-protocol transport.

---

## 8. Behavioral cloning from dense_cnn shards (useful but a rewrite)

BC is a fast signal that Model 2 can fit targets before paying for self-play, but
it is **not free reuse** — it is a **conversion rewrite**:

- dense_cnn shards store plane-oriented inputs and **visit-count policy targets
  over action ids / crop flats**. BC must **reconstruct the graph** (stones,
  candidates, active windows, edges) from the compact raw facts and **map
  dense_cnn's visit-count policy onto Model 2's candidate nodes**.
- **Support mismatch:** dense_cnn searches the full radius-8 legal set; Model 2's
  candidates are the active-window set. Visits on moves outside the
  active-window set (vacuum/dead cells) have **no candidate node**. Handling:
  drop those visits and renormalize over the candidate support, and **log the
  dropped mass** (it should be near-zero if the §4.6 completeness gate holds —
  this is the same measurement). 65-bin value and opp_policy convert directly.
- **Why limited:** BC teaches dense_cnn's policy *as projected onto a different
  move vocabulary and a different architecture* — an initialization aid, not a
  faithful clone. Treat its result as a warm start, then do cold-start/continued
  self-play RL.

Scope BC as a real module (`bc_convert`) with its own tests, not a flag.

---

## 9. Phased roadmap (from-scratch build, modeled on dense_cnn discipline)

Phases 0–4 are CPU-only (no GPU contention with the live dense_cnn run). GPU
phases (5+) wait until the 96×8 run frees the GPU.

**Phase 0 — Delete + scaffold (CPU).** Delete `hexformer_ar` (§1). Create
`packages/hexo_models/hexgt/` mirroring dense_cnn's module set; `constants.py`,
`config.py`; stub `architecture.py`; add the entry point.
*Gate:* installs editable; `load_model_plugin` resolves `hexgt`; forward returns
`{"policy"(dynamic),"value"(N,65),"opp_policy"(dynamic)[,"stvalue_*"]}`.

**Phase 1 — Engine-grounded candidate/window path + bounded edges + validation
(CPU). [GATE]** Implement the shared Rust candidate-set + active-window
enumeration (§4.5) **and the bounded edge construction (§6.3, window-hub, no
cliques)** over the engine `WindowStore`. Run the §4.6 validation on dense_cnn's
recorded games: (1) completeness ≈ 100% (sweep `n=2..8`), (2) candidate/window
size distribution, (3) **node- and edge-count distribution by type (the
no-explosion gate)**.
*Gate:* completeness ≈ 100% (else investigate/adjust); candidate cost acceptable;
**edge count linear in (#nodes + #active-windows) — no super-linear growth.**
**This gates the whole design.**

**Phase 2 — Contract-conformance tests (CPU).** Mirror dense_cnn's test files as
`tests/test_hexgt_*.py`: forward keys/shapes (dynamic policy length = #candidates),
checkpoint round-trip + `sample_buffer` rejection, config unknown-key rejection,
candidate↔priors CSR ordering.
*Gate:* all green on random weights.

**Phase 3 — Dynamic GNN + transformer body (CPU).** Typed message passing (§6.3),
context transformer, candidate cross-attention, optional candidate self-attn,
dynamic policy + 65-bin value heads. Packed-graph collation.
*Gate:* forward on packed synthetic graphs; **D6 equivariance test passes for all
12 elements** (§6.5); overfits one tiny fixed batch.

**Phase 4 — Expand-time graph construction + trainer (CPU). [MAJOR]** Build
`expand_row_to_graph` as the **MVP recompute-at-expand path**: rebuild the
candidate set, active-window tokens, typed nodes, and typed edges per epoch from
the **raw-fact shards** (via the shared Rust function, §4.5/§7 item 1) — **no new
shard `SCHEMA_VERSION`**, shards stay representation-agnostic. Add `HexgtTrainer`
(dense_cnn loss/AMP/clip discipline; dynamic policy CE). Validate targets vs
dense_cnn raw facts for shared rows.
*Gate:* byte-stable graph batches from recompute; targets consistent with engine
facts; a CPU training pass decreases loss. (Accepted tradeoff: per-epoch CPU
recompute cost — measured here so Phase 8 has a baseline to beat.)

**Phase 5 — MCTS integration + torch throughput (GPU). [MAKE-OR-BREAK GATE]**
Rust MCTS encoder emits the graph payload via the zero-copy buffer transport;
Python `inference.evaluate_payload` → `priors_bytes`/`values_bytes`; reuse
dense_cnn's priors validation. Profile pos/s at 512 sims with
`scripts/_profile_selfplay.py` on **torch FP16 (no TRT)**; tune packing/batching.
*Gate (go/no-go):* legal end-to-end self-play; priors validation passes; pos/s
acceptable for self-play (normalized vs dense_cnn by search compute, §10). If
unworkable, reduce model depth or reconsider — honestly.

**Phase 6 — BC warm-start + cold-start RL (GPU, after dense_cnn frees GPU).**
`bc_convert` from dense_cnn shards (§8) → warm start → continued/cold-start
self-play RL. Reuse the scratch-64 autonomy supervisor; watch opening entropy
(`forced_playout_k` / opening-temperature lessons, `MEMORY.md`).
*Gate:* loss decreases on real targets; healthy self-play opening diversity.

**Phase 7 — Head-to-head vs dense_cnn (GPU).** SealBot eval + direct matches
under matched search compute (§10).
*Gate:* Model 2 reaches a defined fraction of dense_cnn's SealBot win-rate at
matched compute — or a clear, honest verdict that it does not.

**Phase 8 — Graph-cache schema bump (deferred perf; gated on the model working).**
*Only once Phases 5–7 prove the model and the graph representation is locked:*
bump the shard `SCHEMA_VERSION` to **cache the precomputed candidate set /
active-window tokens / typed edges** in the shards, replacing per-epoch
recompute. Keep the recompute path as the reference/fallback and assert
cache-vs-recompute byte-equality.
*Gate:* cached expansion is byte-identical to recompute; measured per-epoch CPU
drop vs the Phase 4 baseline. **Not MVP** — skip entirely if recompute is cheap
enough in practice.

---

## 10. Head-to-head evaluation methodology

Re-implement the exact dense_cnn eval harness in `hexgt` (modeled on
dense_cnn) so the comparison is apples-to-apples:

- **Same SealBot eval:** `evaluate_epoch` → `run_match` vs `SealBotPlayer`,
  `sealbot_variant="best"`, `time_limit=0.05`, alternating colors,
  `games_per_epoch=64`; `use_trt=False` for both.
- **Same MCTS:** same batched PUCT + transposition cache + same `search_visits`
  (512 for strength). Only the network + payload encoding differ.
- **Matched compute — lead with this.** Report three axes: (1) **matched search
  visits** (per-search quality); (2) **matched wall-clock self-play budget**
  (penalizes the torch-only/no-TRT throughput honestly vs dense_cnn's TRT
  self-play); (3) **matched param count** (~2.1M, capacity footnote).
- **Direct match:** Model 2 vs dense_cnn (`run_match`, alternating colors, ≥200
  games), win-rate ± Wilson interval.
- **Same opening-diversity controls** (`opening_moves`/`opening_temperature`,
  per-(game,move) seeds, `MEMORY.md`).

---

## 11. Decisions reflected + open questions

**Binding decisions reflected:** delete `hexformer_ar`, build fresh on dense_cnn
discipline (§1); **drop-in PIPELINE compatibility, not shape-matching** (§2);
**dynamic per-candidate policy, no 1681** (§3.4, §6.4); **two layers: model
scores the FULL candidate support (no cap), MCTS nucleus/top-p widens within it
(≈0.95 mass, like dense_cnn)** (§6.6); **bounded edge construction — line
relations via window-node hub, NO same-axis cliques, edges linear in #nodes +
#active-windows** (§6.3); **truly dynamic GNN — no padding/caps, no TRT, torch
FP16** (§6.1); attention-bias formulation
**rejected** as a non-GNN approximation (§4.3); **candidate set = (empty cells in
any active window of either player) ∪ (empty cells within hex-distance ≤ `n` of
any stone), `n` default 2, tunable [2,8], no top-k/no caps** (§4); **tactical
tokens = count-3/4/5 active windows of both colors, no forks, no other types**
(§5); **all stones as nodes, no budget** (§6.2); **BC = conversion rewrite**
(§8); **full-D6 equivariance test** (§6.5); **matched-compute fairness** (§10);
**Rust parity of candidate/window detection across sample-gen and play** (§4.5);
**validation gates** on candidate completeness, candidate size, and **node/edge
counts (no-explosion gate)** (§4.6);
**recompute-at-expand for the MVP, cache-the-graph-when-proven** — shards stay
raw-fact / representation-agnostic with no MVP `SCHEMA_VERSION` bump; the
candidate set, active-window tokens, nodes, and edges are rebuilt per epoch at
expand time; caching into a new `SCHEMA_VERSION` is a later perf phase gated on
the model working (§7 item 2, §9 Phase 8). Accepted MVP tradeoff: per-epoch CPU
recompute in exchange for keeping the representation flexible while it is in flux.

**Open questions:**
1. **Package name:** `hexgt` acceptable?
2. **Aux heads:** confirm dropping WDL/threat/lookahead from the MVP (re-add later
   as private aux off the pipeline contract).
3. **opp_policy / stvalue:** include both auxiliaries in the MVP, or value+policy
   only first?
4. **BC dropped-mass tolerance:** what dropped-visit fraction (from §4.6/§8) is
   acceptable before widening the candidate rule or revisiting?

**Resolved (was open):**
- **Graph in shard vs recompute-at-expand → DECIDED: recompute-at-expand for the
  MVP; cache into a new `SCHEMA_VERSION` only once the model is proven** (§7
  item 2, §9 Phase 8). Rationale: keep shards representation-agnostic while the
  graph rep is still being proven; the schema bump is a deferred optimization.

---

## 12. Readiness review — specification gaps before implementation

Honest pass over the plan. Each item: what's underspecified, and whether it
**blocks Phase 0** (scaffold/stub), blocks a **later phase**, or is **decide-later**.
Phase 0 only needs: the package skeleton, config stub, and a forward stub
returning the right output **keys** — so most gaps below are *not* Phase-0
blockers, but several must be nailed before the phase that depends on them.

**A. Exact node feature vectors (dims + encodings).** §6.2 lists feature
*contents* but not concrete dimensions, normalization, or how categoricals
(owner, phase, axis, count) are encoded (one-hot vs embedding) or how relative
coords are featurized (raw q/r/s vs sin/cos). *Blocks Phase 3* (model body) and
the Phase 4 expand step; **not** Phase 0. Decide alongside the first GNN
implementation; lock before training.

**B. Two-stone turns → policy/MCTS action mapping + ActionId.** The engine turn
is autoregressive (FirstStone → SecondStone, `state.rs:46-56`), but the plan does
not state whether the model/MCTS treats each **single stone** as one action
(matching the engine's per-placement legality, simplest, and what dense_cnn's
per-move CSR implies) or a two-stone macro-action. Almost certainly per-single-
stone (the engine exposes single-placement legality and the priors contract is
per-move). **This must be confirmed explicitly** — it affects the policy target,
the value perspective sign across the two half-moves, and `opp_policy`
("opponent's next decision" must be defined w.r.t. the two-stone turn boundary).
ActionId is the engine's `pack_coord` u32 (`legal.rs:24-35`) — already settled.
*Blocks Phase 4* (sample-gen/targets); **decide before Phase 4**, not Phase 0.

**C. Value & short-term-value target definitions on the new rep.** §6.4 reuses
dense_cnn's 65-bin head, but the **target-construction** (winner→±1 value,
`opp_policy` = future opponent decision, EMA short-term value per horizon) is
dense_cnn's `finalize_game_samples` logic and must be re-derived for two-stone
turns and the dynamic candidate support (esp. how `opp_policy` maps onto *this*
position's candidate nodes vs the opponent's later candidate set). *Blocks
Phase 4.* Tied to (B).

**D. D6 applied to typed edges (not just coords).** §6.5 says node/edge identity
is permutation-invariant and axes relabel, but does not spell out the **axis
permutation map** under each of the 12 group elements (which of Q/R/QR maps where
under each rotation/reflection) nor that edge *endpoints* are preserved while
edge *type* (if axis-typed) must relabel. The equivariance test depends on this
being exact. *Blocks Phase 3* (the equivariance test). Specify the axis-relabel
table when implementing `d6.py`.

**E. BC conversion specifics.** §8 scopes the rewrite but leaves open: the exact
mapping from dense_cnn's crop-flat/action-id visit targets onto candidate nodes,
the renormalization after dropping out-of-candidate visits, and the
dropped-mass tolerance (open question #4). *Blocks Phase 6 only* (BC is a
warm-start, late). Decide after the Phase-1 coverage numbers are in.

**F. Optimizer / LR / loss-weight starting points.** The plan says "same
discipline as dense_cnn" but gives no concrete starting values for the new rep
(dense_cnn defaults: AdamW, lr 1e-3, wd 1e-4, policy 1.0 / value 1.0 / opp 0.25 /
stvalue 0.25). A transformer/GNN typically wants a lower LR + warmup than a CNN.
*Blocks Phase 4/6 training*, not Phase 0. Start from dense_cnn weights, add
warmup; tune empirically.

**G. Eval/MCTS config for the fair comparison.** §10 fixes most of it
(`search_visits=512`, SealBot best-50ms, alternating colors, `use_trt=False`),
but the **nucleus widening threshold** (§6.6) and `c_puct`, dirichlet/temperature
schedule, and `max_actions` for Model 2 are unspecified. They must be **identical
between self-play and eval** and comparable to dense_cnn's. *Blocks Phase 5/7.*
Default to dense_cnn's selfplay config values; document any deviation.

**H. GNN/transformer hyperparameters (depth/width).** §3.6 gives a target param
budget (~2.1M) and rough shape (`token_dim≈128`, `L_gnn 2–3`, `L_ctx 2–4`) but
not final layer counts, aggregation (mean vs sum vs attention-pool), or the
context-attention sparsity. *Blocks Phase 3.* Pick to hit the param budget after
the Phase-1 cost numbers; iterate.

**I. Packed-graph collation contract.** §6.1/§6.3 name PyG-style packing +
block-diagonal masking but don't define the concrete batch tensors (the
`graph_id`/segment layout, per-type node/edge index tensors, the CSR mapping from
candidate nodes back to `legal_row_offsets` for `priors_bytes`). *Blocks Phase 3
(collation) and Phase 5 (Rust payload).* This is the interface between the Rust
encoder and the torch model — specify it before Phase 5.

**J. Transposition-cache key on the dynamic rep.** dense_cnn's Rust MCTS keys the
cache by `hash_state`; the plan reuses the framework but doesn't confirm the key
is the engine state hash (it should be — the graph is a deterministic function of
state). *Decide-later*, low risk; confirm in Phase 5.

**Not gaps (already settled):** candidate-set rule + `n` (§4), tactical-token
taxonomy (§5), delete-and-rebuild + pipeline interfaces (§1–2), recompute-at-
expand (§7/Phase 8), no-TRT torch inference (§6.1), edge-cardinality bounds
(§6.3), the validation gates (§4.6).

**Phase-0 blockers: none** — scaffolding can proceed. **Earliest hard blockers**
are (B)+(C) for Phase 4 and (A)+(D)+(H)+(I) for Phase 3; resolve those two
clusters first once implementation starts.

---

## Appendix A — engine + pipeline file:line index

Engine rules:
- Win / `WINDOW_LEN=6` / axes: `packages/hexo_engine/rust/src/tactics.rs:14-53`
- **Active window** definition: `tactics.rs:172-186`
- Threat (count≥4): `tactics.rs:189-198`; popcount count: `tactics.rs:133-136`
- Win window: `tactics.rs:206-208`; blocked-not-threat test: `tactics.rs:620-636`
- WindowStore enumeration: `tactics.rs:154-156,378,386-395`
- Legal radius 8: `packages/hexo_engine/rust/src/legal.rs:10-11,124-128`
- Turn phases / two-stone turns / win check: `packages/hexo_engine/rust/src/state.rs:9-10,46-56,304-330`
- No draws / unbounded board: `state.rs:64-71,98`; `coord.rs:1-9`
- hex_distance / coords_within_radius: `coord.rs:77-96`
- Opening forced to (0,0): `rules.rs:16-23`, `state.rs:224-228`

Pipeline contracts:
- dense_cnn value bins: `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/losses.py:20-30`
- dense_cnn checkpoint: `.../dense_cnn/checkpoints.py:23-105`
- dense_cnn replay window/shuffle: `.../dense_cnn/replay.py:523-535,615-755`
- dense_cnn config: `.../dense_cnn/config.py:183-318`
- dense_cnn plugin/entry point: `.../dense_cnn/plugin.py:27-119`, `packages/hexo_models/pyproject.toml:17-19`
- Rust↔Python evaluator payload + zero-copy buffer: `.../dense_cnn/rust/src/mcts_eval.rs:48-100,315-390`
- ModelPlugin protocol: `packages/hexo_train/python/hexo_train/registry.py:24-103`
- hexformer (to delete) Rust include: `packages/hexo_models/rust/src/lib.rs:7-9,24-29`
