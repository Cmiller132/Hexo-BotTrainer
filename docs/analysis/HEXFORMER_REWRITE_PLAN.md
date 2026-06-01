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
  fixed shapes, no `max_candidates`, no fallback logits, no top-k anywhere.
  Variable node/edge counts are first-class (§3).
- **The policy is dynamic and per-legal-move (pointer/CSR), not a 1681 dense
  crop.** The head emits one logit per candidate node; this maps *directly* onto
  the Rust MCTS priors contract, which is already per-legal-move CSR
  (`legal_row_offsets` + `priors_bytes`, §2.6).
- **The candidate set is defined by a single engine-grounded rule — no radius,
  no hyperparameter:**

  > **`candidate_set(position) = { every EMPTY cell that lies in ≥1 ACTIVE
  > window of EITHER player }`**

  where an *active window* is the engine's existing concept — a length-6 line
  window containing stones of exactly one player (`tactics.rs:183-186`). This
  covers threat-completion/block cells (count 4/5), developing cells (count 1–3),
  own extensions and opponent blocks, and naturally **excludes** vacuum cells
  (windows with zero stones) and dead cells (windows contested by both colors).
  See §4.
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

### 4.1 Definition (engine-grounded, no hyperparameter)

```
candidate_set(position) = { every EMPTY cell that lies in ≥1 ACTIVE window
                            of EITHER player }
```

Concretely in Rust: iterate `board().windows().entries()`, keep those with
`is_active()` (either color, any count ≥ 1), collect their `empty_cells()`,
dedupe. This is exactly the set of empty cells that could **extend** one of the
current player's lines or **block** one of the opponent's. It:

- **includes** threat-completion and must-block cells (count 4/5 windows),
  developing cells (count 1–3 windows), own extensions, and opponent blocks;
- **excludes** vacuum cells (cells whose only windows have zero stones) and dead
  cells (cells all of whose windows are contested by both colors → no active
  window).

**No radius, no `n`, no top-k, no `max_candidates`, no fallback logits.** The
count is dynamic per position.

### 4.2 Why prune at all (model tractability, not legality)

The prune exists for **model tractability, not MCTS legality**. dense_cnn's conv
policy head outputs all 1681 crop logits "for free" from a fixed-size feature
map. A token GNN/transformer instead pays ~O(N²) attention and per-node message
cost in the **number of candidate tokens** N. The active-window filter bounds N
to the *connection-relevant* empty cells — the only cells that can extend or
block a six-in-a-line — which is exactly the move vocabulary a 6-in-a-row game
actually uses. Because the dynamic policy is defined over these candidate tokens,
**the candidate set is also exactly what MCTS expands at that node** (priors are
emitted per candidate; the Rust search children = candidate tokens).

This is sound for Hexo specifically: a move that lies in **no** active window
neither extends any existing one-color line nor blocks any opponent line, so it
cannot create or defend a six — it is strictly a tempo-losing/“vacuum” move in a
pure connection game with no captures and no territory. Pruning such moves
removes only moves that cannot participate in the win condition.

### 4.3 Relationship to the old radius idea (dropped)

A hex-distance `n` radius around stones is **dropped entirely**. Note *why a pure
radius would have been unsafe*: with win length 6, a window-completing cell can
sit up to 5 line-cells from the nearest stone of that window, so a small radius
(e.g. n=3) could exclude a legal threat-completion/block move. The active-window
rule has **no such failure mode** — every completable line's empty cells are
included by construction, at any distance, because they belong to an active
window. This is strictly safer than any radius and needs no tuning.

> The attention-bias / fixed-shape padded formulation considered earlier is a
> **non-GNN approximation and is rejected**: it cannot represent the truly
> dynamic, unbounded graph structure the user wants and would reintroduce caps/
> padding. It is mentioned only to record that it was evaluated and declined.

### 4.4 Opening special case

On an **empty board there are no stones, hence no windows, hence an empty
candidate set.** Handling:

- **Move 1** is the engine-forced `Opening` placement at `(0,0)`
  ([rules.rs:16-23](packages/hexo_engine/rust/src/rules.rs:16), [state.rs:224-228](packages/hexo_engine/rust/src/state.rs:224)) — a single candidate; no model choice
  needed. The pipeline can hard-code it (matching engine legality).
- **Move 2 onward:** once the center stone exists, the 18 windows through `(0,0)`
  are active (count 1, owner Player 0), so `candidate_set` is non-empty — it is
  the empty cells of those 18 length-6 lines through the center (the natural
  opening region). No special code needed beyond move 1.
- **Safety net:** if any non-terminal position ever yields an empty candidate set
  (should only be the pre-opening empty board), fall back to the engine's legal
  move list for that node and log it loudly (no silent divergence between
  candidates and legality).

### 4.5 Rust parity (training ↔ play)

The candidate-set construction **and** the active-window detection must run in
**one shared Rust path** used by **both** sample-gen and live MCTS, so the move
vocabulary at a node is identical in training data and in play. (Mismatch would
train the policy over a different support than search expands.) This is a single
function over the engine `WindowStore`, called from both `sample_gen` and the
MCTS encoder.

### 4.6 Validation gates (before locking the design)

Replay dense_cnn's recorded games (`runs/.../selfplay/*.hxr` / shards,
read-only) and measure:

1. **Completeness/safety:** fraction of dense_cnn's **actually-played** moves and
   **MCTS-visited** moves that fall inside `candidate_set`. **Target ≈ 100%.**
   (dense_cnn searches the full radius-8 legal set, so this directly tests
   whether the active-window prune ever excludes a move a strong player used. If
   < ~100%, investigate the excluded moves — they should be vacuum/dead by
   construction; a genuine miss is a design red flag.)
2. **Cost distribution:** the **size distribution** of `candidate_set` across
   game phases (opening / midgame / endgame): report median, typical, and p95
   candidate counts, plus active-window-token counts. This sets the GNN's
   per-node cost and the realistic batch sizes for the throughput phase.

Both are **gates**: (1) proves the prune is complete/safe; (2) proves the dynamic
cost is bounded enough to be trainable and searchable at acceptable throughput.

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

Relationship to the candidate filter: the **candidate-cell filter uses ALL
active windows (count ≥ 1)**; the **window tokens are the higher-count (3/4/5)
subset**. Count-1/2 active windows still contribute their empty cells as
candidates but are not emitted as tactical tokens (they carry little structure
beyond locality, which the candidate node's own features already express).

**No forks / double-threats are computed.** A cell sitting in multiple active
windows *is* a fork; the GNN/transformer learns this implicitly because that
candidate node attends to (is edged to) multiple window tokens (§6.3). **No
other window types** (broken/gapped is already covered by popcount §3; no
intersection/VCF/dead-line/edge tokens). **No top-k** — all count-3/4/5 active
windows are emitted; the count is dynamic.

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

### 6.3 Typed edges (the heterogeneous graph)

- **stone↔stone** hex-adjacency and same-axis-line membership (group structure).
- **window↔stone**: a window node connects to the (one-color) stones it contains.
- **window↔candidate**: a window node connects to its empty cells — this is the
  edge that lets a candidate “see” every threat/extension it participates in
  (and thus learn forks implicitly).
- **candidate↔stone**: a candidate connects to nearby stones (local context).
- **side/goal↔all**: global broadcast.

Message passing is type-conditioned: `m_{j→i} = φ_{edge_type}(h_j, h_i,
rel_coord_{ij})`, aggregated per target node (segment scatter) over `L_gnn`
rounds. Then a transformer block set provides global attention (context
self-attention over {side, stone, window}; candidate→context cross-attention;
optionally one candidate↔candidate sparse self-attention), all masked
block-diagonally per graph in a packed batch.

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

---

## 7. Sample-gen + training rewrite (scope, honestly)

**Emulate dense_cnn's discipline** (raw-fact columnar NPZ + offsets +
`SCHEMA_VERSION`, power-law replay window, md5 split, batch-aligned shards, JSON
sidecars, per-epoch D6-at-read, checkpoint hygiene, config/plugin/test patterns).

**New, largely-rewritten work:**
1. **Shared Rust candidate/active-window path** (§4.5): one function over the
   engine `WindowStore` producing the candidate set + the count-3/4/5 window
   tokens, called by both sample-gen and live MCTS.
2. **Rust sample-gen:** emit typed-node raw facts (all stones, candidates, active
   windows) + typed edges. Likely a new `SCHEMA_VERSION` (graph fields) or
   recompute the graph at expand time from the raw facts the compact schema
   already stores (`stones_qr`, `legal_ids`, plus windows recomputable from
   stones) for the MVP.
3. **Expand step (`expand_row_to_graph`, new):** compact row + D6 symmetry →
   typed node tensors + typed edge lists + dynamic policy/opp targets + 65-bin
   value/stvalue targets. Replaces dense_cnn's `expand_*_to_planes`.
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

**Phase 1 — Engine-grounded candidate/window path + validation (CPU). [GATE]**
Implement the shared Rust candidate-set + active-window enumeration (§4.5) over
the engine `WindowStore`. Run the §4.6 validation on dense_cnn's recorded games:
(1) completeness ≈ 100%, (2) candidate/window size distributions.
*Gate:* completeness ≈ 100% (else investigate/adjust); cost distribution
acceptable for training/search. **This gates the whole design.**

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

**Phase 4 — Sample-gen + expand + trainer (CPU). [MAJOR]** Rust typed-node +
typed-edge sample-gen (§7); `expand_row_to_graph`; `HexgtTrainer` (dense_cnn
loss/AMP/clip discipline; dynamic policy CE). Validate targets vs dense_cnn raw
facts for shared rows.
*Gate:* byte-stable graph batches; targets consistent with engine facts; a CPU
training pass decreases loss.

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
**dynamic per-candidate policy, no 1681** (§3.4, §6.4); **truly dynamic GNN — no
padding/caps/top-k, no TRT, torch FP16** (§6.1); attention-bias formulation
**rejected** as a non-GNN approximation (§4.3); **candidate set = empty cells in
any active window of either player, no radius/no `n`/no top-k** (§4); **tactical
tokens = count-3/4/5 active windows of both colors, no forks, no other types**
(§5); **all stones as nodes, no budget** (§6.2); **BC = conversion rewrite**
(§8); **full-D6 equivariance test** (§6.5); **matched-compute fairness** (§10);
**Rust parity of candidate/window detection across sample-gen and play** (§4.5);
**validation gates** on candidate completeness and size distribution (§4.6).

**Open questions:**
1. **Package name:** `hexgt` acceptable?
2. **Aux heads:** confirm dropping WDL/threat/lookahead from the MVP (re-add later
   as private aux off the pipeline contract).
3. **opp_policy / stvalue:** include both auxiliaries in the MVP, or value+policy
   only first?
4. **Graph in shard vs recompute-at-expand:** cache edges/windows in a new
   `SCHEMA_VERSION`, or recompute from raw facts at expand time for the MVP?
5. **BC dropped-mass tolerance:** what dropped-visit fraction (from §4.6/§8) is
   acceptable before widening the candidate rule or revisiting?

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
