# Hexformer Rewrite Plan — GNN + Transformer Hybrid (Model 2)

**Status:** Design / planning only. No implementation, no pipeline changes. The
dense_cnn 96×8 run is live on the GPU — this plan must not compete for GPU and
nothing here is to be run during that run.

**Date:** 2026-06-01 (revised after user direction)
**Author:** read-only analysis pass
**Reference baseline / template to emulate:** `hexo_models.dense_cnn` (Model 1),
96ch×8blk, currently beating SealBot best-50ms (~92% @ e17 — see `MEMORY.md`).

---

## 0. TL;DR / Executive summary

- **Decision (binding): delete `hexformer_ar` and build Model 2 from scratch.**
  `hexformer_ar` is completely untested (no `runs/hexformer_ar` artifacts, never
  trained), bloated, and is **not** a sound basis. We do **not** carry its model
  or its scaffolding forward as the foundation. The build is **modeled on
  `dense_cnn`**, the proven reference — its package structure, config system,
  `ModelPlugin` pattern, trainer/replay/checkpoint discipline, MCTS integration,
  and test discipline are the template to emulate. `hexformer_ar` is referenced
  only as a *cautionary example* and, at most, as a non-authoritative source for
  a couple of small pure-math helpers (hex-D6 group, axial/cube coordinate
  packing) that we will re-derive and re-test from scratch rather than import.
- **The model is fundamentally a GNN.** Model 2 is the user's hybrid: a **typed
  heterogeneous GNN local encoder** (placed stones / legal moves / tactical
  windows / side+goal tokens, with typed edges) → **transformer global
  attention** → **dense_cnn-style heads** (65-bin value + per-move policy +
  opponent-policy aux + short-term value). The explicit typed GNN is a
  **first-class component**, not an implementation detail. A fixed-shape,
  padded, *attention-bias* formulation is presented as an **optimization option**
  for making the GNN performant and TRT-exportable — evaluated in the throughput
  phase — **not** as a substitute that removes the GNN (§3, §4.1).
- **This is a major sample-generation and training rewrite — say so plainly.**
  We emulate dense_cnn's *data discipline* (raw-fact NPZ shards, power-law
  replay window, on-disk shuffle, per-epoch D6 at read, byte-identical schema
  versioning, checkpoint hygiene). But the **graph representation requires
  substantial NEW work**: graph/token + typed-edge construction (the Rust
  sample-gen + the expand step), variable-size graph collation/batching, and a
  new trainer. This is **not** "reuse the replay code wholesale" — it is "adopt
  the discipline, rewrite the sample-gen and training for the graph rep" (§5,
  §6).
- **The single make-or-break risk is inference throughput in the 512-sim MCTS
  hot loop.** dense_cnn fought to ~58 pos/s with a fixed-shape CNN + TensorRT
  FP16. A message-passing GNN is harder to batch and may not export to TRT. The
  throughput phase (Phase 5) is a hard go/no-go gate that decides the model's
  final form and viability.
- **Drop-in compatibility is a hard requirement.** Model 2 must honor the exact
  contracts dense_cnn already satisfies (forward output keys/shapes, the Rust↔
  Python MCTS evaluator callback, checkpoint payload, plugin protocol, SealBot
  eval harness) so the existing machinery and the head-to-head comparison "just
  work." These contracts are catalogued in §2 and matched in §3.4 / §9.

---

## 1. What the current `hexformer_ar` is — and why it is the wrong basis

Package: `packages/hexo_models/hexformer_ar/` — Python in
`python/hexo_models/hexformer_ar/`, Rust in `rust/src/`. ~3,200 lines Python +
~3,500 lines Rust.

### 1.1 Architecture (`architecture.py`)

A sparse hybrid: local hex-CNN encoder → typed-token GraphGPS stack →
candidate-pointer policy. The "AR" (autoregressive) in the name is vestigial —
`forward()` is single-shot.

- **`LocalHexEncoder`** ([architecture.py:71-84](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:71)):
  `HexConv2d` (corners masked for the hex 6-neighborhood) → `GatedHexBlock`
  stack → avg-pool → `Linear` to `token_dim`.
- **Typed token assembly** ([architecture.py:221](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:221)): concat
  `[global, local_window, candidate, stone, window]` tokens, each tagged with one
  of 5 learned `type_embedding` rows.
- **`GraphGPSBlock` × `gps_layers`** ([architecture.py:87-164](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:87)):
  coordinate-conditioned local MLP message + **explicit edge message passing via
  a per-batch-element Python `for` loop with `index_add_`** (`_edge_aggregate`,
  [architecture.py:134-164](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:134)) + full `nn.MultiheadAttention` + FFN.
- **Pointer policy** ([architecture.py:240-264](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:240)): one logit per
  candidate, masked to legal.

`forward()` returns `policy_logits/policy [B,C]`, `opp_policy [B,C]`, `wdl_logits
[B,3]`, `distance [B]`, `threat_logits [B,C,6]`, `rz_logits [B,C]`,
`lookahead_{1,2,4,8} [B,3]`. **This breaks the dense_cnn contract**: 3-class WDL
instead of the 65-bin distributional value; variable `(B,C)` pointer policy
instead of the trainer-side `(N,1681)`; and a different aux-head set.

### 1.2 Why it is a bad basis (the verdict)

- **Never trained / wholly unvalidated.** No `runs/hexformer_ar` artifacts;
  `MEMORY.md` records dense_cnn iteration only. The unused `HexformerOutputs`
  dataclass ([architecture.py:15-23](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/architecture.py:15)) and vestigial "AR" name signal
  design drift never shaken out by real training.
- **Bloated and structurally off-pattern.** The GNN is a per-batch-element
  Python loop (`_edge_aggregate`) — slow and not exportable. Replay is an
  in-memory **zlib+JSON** buffer ([samples.py:42-53](packages/hexo_models/hexformer_ar/python/hexo_models/hexformer_ar/samples.py:42)) and the
  checkpoint **persists `sample_buffer`** — the *opposite* of dense_cnn's
  discipline (which rejects legacy `sample_buffer` payloads).
- **Diverged contracts.** Heads, value representation, and policy shape all
  diverge from the proven dense_cnn contract, so "drop-in comparison" would
  require undoing its choices anyway.

**Verdict: DELETE `hexformer_ar`. Build Model 2 fresh, modeled on dense_cnn.**
We do not copy its package, plugin, config, trainer, samples, or Rust crate as a
foundation. The only artifacts we may consult (not import) are the hex-D6 group
math and the axial/cube coordinate packing — and even those we re-derive and
re-test from scratch in the new package, because the whole point is a clean,
tested build on the proven pattern. See §7 for what "delete" means in practice
(remove the package dir, its entry point, and its `#[path]` include).

---

## 2. The dense_cnn contracts Model 2 must honor (for drop-in + fair comparison)

dense_cnn is both the **template to emulate** and the set of **contracts to
match**. Quote-accurate as of this writing.

### 2.1 Forward output contract (`dense_cnn/architecture.py`)

`Model1Network.forward(x)` → `dict[str,Tensor]`
([architecture.py:202-211](packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/architecture.py:202)):

```python
outputs = {
    "policy":     self.policy_head(features),     # (N, 1681)  == (N, BOARD_AREA)
    "value":      self.value_head(features),      # (N, 65)    == (N, VALUE_BINS)
    "opp_policy": self.opp_policy_head(features), # (N, 1681)
}
for horizon, head in self.short_term_value_heads.items():
    outputs[f"stvalue_{horizon}"] = head(features)  # (N, 65) each; key is int horizon
```

- `policy`/`opp_policy`: `(N, 1681)`, `BOARD_AREA = 41*41` (`constants.py:10`).
- `value`: `(N, 65)` — **65-bin distributional head** over fixed support
  `torch.linspace(-1.0, 1.0, 65)` (`losses.py:20-23`); decode = softmax·bins.
- `forward_policy_value(x)` ([architecture.py:213-220](packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/architecture.py:213)) returns only
  `{"policy","value"}` — the inference path the Rust MCTS uses.

**Param count** is config-derived (defaults 96/6), not asserted. 96×6 ≈ ~1.6M,
96×8 ≈ ~2.1M by hand estimate. The "~2.6M" in the brief is **not in code**
(only the removed FC-head's "~5.6M params/head" comment). Size Model 2 to the
*running* 96×8 baseline (~2.1M), measured by `sum(p.numel())`.

### 2.2 Input representation

`constants.py`: `BOARD_SIZE=41`, `BOARD_AREA=1681`, `INPUT_CHANNELS=13`,
`VALUE_BINS=65`. 13 fixed planes (`constants.py:16-28`). Geometry
(`geometry.py`): crop center = rounded mean axial coord; `coord_to_flat =
row*41 + col`, `half=20` — the **1681 action-space mapping**; out-of-crop facts
silently dropped. `ActionId u32_i16_pair` (`d6.py:19,41-59`): `_COORD_OFFSET =
1<<15`; pack `((q+32768)<<16) | (r+32768)`.

### 2.3 Trainer / loss (`dense_cnn/trainer.py`, `losses.py`)

`DenseCNNTrainer(model, config, optimizer)`; optimizer is **AdamW**. Loss
`model1_loss` (`losses.py:114-140`):

```
total = policy_weight*CE(policy) + value_weight*binnedCE(value)
      + opp_policy_weight*CE(opp_policy)
      + short_term_value_weight*binnedCE(stvalue_h, mask=stvalue_h_mask)
```

Defaults `policy=1.0, value=1.0, opp_policy=0.25, short_term_value=0.25`. Step
consumes `batch: dict[str,Tensor]` with `"input" (N,13,41,41)` + target keys;
AMP via `GradScaler`; optional `clip_grad_norm_`.

### 2.4 Compact NPZ schema (`compact_io.py`, `replay.py`) — discipline to emulate

`COMPACT_SCHEMA_VERSION = 1` (`compact_io.py:33`). One `.npz` = N rows,
columnar, varlen fields as concatenated data + `int64` offsets `(N+1)`. **Raw
game facts, not planes** — `turn_index`, `current_player`, `phase`,
`center_q/r`, `value`, `first_q/r/present`, `stvalue`/`stvalue_mask`,
`stones_qr`(+`stones_off`,`stones_owner`), `legal_ids`(+`legal_off`),
`hist_qr`(+`hist_owner`,`hist_idx`,`hist_off`),
`own_hot_qr`/`opp_hot_qr`/`last_hot_qr`(+offs), `pol_act`(+`pol_w`,`pol_off`),
`opp_act`(+`opp_w`,`opp_off`). `expand_shard_to_arrays`
(`compact_io.py:243-322`) expands row `i` under `symmetries[i]` to **dense
planes**.

> **What we emulate vs rewrite (be honest):** the *schema philosophy* (raw facts
> + offsets + version), the *replay window/shuffle/split* mechanics
> (`replay.py:523-755`), and the *per-epoch D6-at-read* are worth emulating and
> can be closely mirrored. But the **expansion step is a from-scratch rewrite**
> (`expand_*_to_planes` → `expand_*_to_graph`), the **Rust sample-gen** that
> emits typed-edge facts is new (§6.1), and the schema likely **gains
> graph-specific fields** (a version bump). See §5 for the full scope.

### 2.5 Samples / D6 / checkpoint / config / plugin (dense_cnn patterns)

- **Samples** (`samples.py`): Rust authors state-derived facts; Python attaches
  search policy + finalizes (`finalize_game_samples`: winner value, future-opp
  policy, EMA short-term value). `expand_sample` applies D6 then builds tensors.
  `symmetry_drops_support` detects square-crop spill.
- **D6** (`d6.py`): `D6_SIZE=12`; applied at read/expand time via a
  per-(run,epoch) symmetry vector, re-randomized each epoch. Square crop is not
  D6-closed → identity fallback for spilling rows.
- **Checkpoint** (`checkpoints.py`): `{"model","model_state","optimizer_state",
  "train_state","epoch","metadata"}`. Rejects a non-None `sample_buffer`,
  incompatible `model_state`, missing `.txt` pointer. Supports `.txt`
  indirection.
- **Config** (`config.py`): `parse_model1_config`; frozen dataclass sections;
  per-section unknown-key rejection; per-scalar coercion; no range validation.
- **Plugin / registry** (`registry.py:24-103`, `dense_cnn/plugin.py:27-119`):
  `ModelPlugin` Protocol — `name`, `build_model(game_spec, config)`,
  `training_component_overrides(*, defaults, config, shared, model) ->
  ComponentOverrides`; optional `calibrate_performance`, `generate_selfplay`,
  `evaluate_epoch`. Resolution by `module` → `entry_point` → `name` over the
  `"hexo_train.models"` group.

### 2.6 The Rust ↔ Python MCTS evaluator callback — the load-bearing interface

Rust owns the search loop and calls back per leaf batch:
`evaluator.call1((payload,))` ([mcts_eval.rs:335](packages/hexo_models/dense_cnn/rust/src/mcts_eval.rs:335)).

**Rust → Python payload dict** (for dense_cnn): `"inputs"` = a `PlaneBuffer`
`#[pyclass]` exposing the **Python buffer protocol** over a `Vec<half::f16>`
(**zero-copy**; Python does `torch.frombuffer(...).reshape(shape)`), `"shape"` =
`(N,13,41,41)`, `"legal_flat_indices_bytes"` = i64, `"legal_row_offsets"` = N+1
i64 CSR. **Python → Rust:** `"values_bytes"` (N f32, clamped `[-1,1]`),
`"priors_bytes"` (f32, **one per legal flat index**, row-major in
`legal_row_offsets` order; Rust validates finite/nonneg/unique/positive-mass and
normalizes).

> **Key for Model 2:** the priors contract is **already per-legal-move (CSR)**.
> dense_cnn computes dense `(N,1681)` then gathers; a **per-legal-move GNN/pointer
> policy maps directly onto `priors_bytes`** with no dense intermediate. Model 2
> reuses this *protocol shape* (the buffer-protocol payload + CSR offsets) but
> ships a **graph/token payload** instead of planes (§6.2).

### 2.7 TensorRT (`dense_cnn/trt_backend.py`)

`build_trt_forward(model, ...)` → drop-in `forward_policy_value`. STRONGLY_TYPED
FP16 baked into exported ONNX; engine built per-epoch in an isolated subprocess
from folded inference weights; correctness-gated (`policy_argmax_match ≥ 0.90`,
decoded `value_max_err ≤ 0.05`), else fail-loud (opt-in torch fallback). **TRT
is self-play only**; eval uses `use_trt=False`.

---

## 3. Model 2 — the GNN + transformer hybrid, concretely

Working name: **`hexgt`** (Hex Graph-Transformer). The model is **fundamentally a
typed heterogeneous GNN**; the transformer provides global attention on top; the
heads reproduce dense_cnn's contract.

### 3.1 Architecture: explicit typed GNN is first-class

```
inputs: placed stones + all legal moves + selected tactical-window tokens
        + side/goal tokens   (typed nodes with relative axial/cube coords)
   │
   ▼
[1] TYPED HETEROGENEOUS GNN local encoder        ← FIRST-CLASS, the core of the model
   │   node types: {stone, legal_move, window, side/goal}
   │   typed edges (see §3.3): adjacency, line-membership, candidate-of, recency,
   │     side-context. Each edge type has its own message function.
   │   L_gnn rounds of typed message passing (local graph reasoning).
   ▼
[2] CONTEXT TRANSFORMER over encoded nodes        ← global attention
   │   full self-attention across {side/goal, stone, window} context nodes
   │   (KataGo-style global mixing the local GNN can't reach).
   ▼
[3] LEGAL-MOVE CROSS-ATTENTION to context         ← "legal moves attend to state"
   │   legal-move nodes are queries; context nodes are keys/values.
   ▼
[4] (optional) ONE SPARSE LEGAL-MOVE SELF-ATTN    ← legal moves attend to each other
   │   masked to spatial neighborhoods.
   ▼
[5] HEADS (dense_cnn contract — §3.4):
      policy  (per legal move → priors_bytes; scattered to (N,1681) for training)
      value   (N,65)   ·  opp_policy (per legal move)  ·  stvalue_<h> (N,65)
```

The GNN in **[1]** is the model's identity and is kept explicit. **[2]–[4]** are
the transformer half of the hybrid. The decision about *how* the typed message
passing is implemented in the inference hot path (true scatter/gather message
passing vs the fixed-shape attention-bias realization) is an **optimization
question resolved in Phase 5**, not a reason to drop the GNN — see §4.1.

### 3.2 Node/token types and features

| type | source (compact facts) | budget (initial, measure) | features |
|---|---|---|---|
| **side/goal** (1–2) | `current_player`, `phase`, global counts | 1–2 | side-to-move, phase one-hot, stone counts, move number, goal encoding |
| **stone** | `stones_qr`+`stones_owner` | ≤ `max_stones` (~256) | owner, recency (`hist_idx`), hot flags, rel-coord |
| **legal-move** | `legal_ids` | ≤ `max_candidates` (~384) | rel-coord, ring/distance, local neighbor occupancy, tactical flags |
| **tactical-window** | derived in Rust (§3.3) | ≤ `max_windows` (~64) | window axis, own/opp/empty counts along the line, open-ends |

Coordinates are **relative axial/cube anchored at the crop center** (re-derived
coordinate helpers, re-tested). "Tactical window" = a fixed-length axis-aligned
line segment (engine win-length) through/near recent or contested cells, scored
by `(own_count, opp_count, open_ends)` and top-k kept. Windows are **derived**,
so for the MVP they can be recomputed at expand time from `stones_qr`+`legal_ids`
(no schema field) or cached in the shard (schema bump — §6.1).

### 3.3 Typed edges (the heterogeneous graph)

Edge types (each with its own learned message function in [1]):

- **hex-adjacency**: stone↔stone / stone↔legal_move / legal_move↔legal_move when
  within the hex 6-neighborhood (locality).
- **line-membership**: window↔stone and window↔legal_move for cells on that
  window's line (threat structure).
- **candidate-of**: legal_move↔stone for the moves adjacent to a stone group.
- **recency**: stone↔stone ordered by `hist_idx` (temporal structure).
- **side-context**: side/goal↔everything (global broadcast).

Edges are constructed in Rust at sample-gen and/or at expand time (§6.1). The
message-passing scheme is type-conditioned: `m_{j→i} = φ_{edge_type}(h_j, h_i,
rel_coord_{ij})`, aggregated per target node (mean/sum), per round.

### 3.4 Heads — EXACT dense_cnn contract mapping

| dense_cnn head | Model 2 analog | shape | notes |
|---|---|---|---|
| `policy (N,1681)` | per-legal-move logit → **scatter** to `coord_to_flat` for the training target; **gather** per-legal-move for inference `priors_bytes` | `(N,1681)` train / `(N,n_legal)` infer | one head, two read-outs |
| `value (N,65)` | 65-bin head on pooled side/goal token | `(N,65)` | **identical**; reuse the `linspace(-1,1,65)` support + `binned_value_loss` math |
| `opp_policy (N,1681)` | per-legal-move opp head, same scatter | `(N,1681)` | from compact `opp_act`/`opp_w` |
| `stvalue_<h> (N,65)` + mask | per-horizon 65-bin head | `(N,65)` | from compact `stvalue`/`stvalue_mask` |

**Drop** the hexformer-specific `wdl/distance/threat/rz/lookahead` heads from the
MVP contract (they break value-binning + MCTS value reuse). Threat/relevance can
return later as *private* auxiliaries off the shared contract.

**Policy scatter detail:** for the training target, scatter each legal-move logit
to `coord_to_flat(action_coord, center)` and set non-legal flats to `-inf`
(dense_cnn's legal mask); compute `soft_cross_entropy` exactly as dense_cnn. For
inference, skip the scatter and emit per-legal-move softmax as `priors_bytes` in
`legal_row_offsets` order. Same head, both read-outs — byte-compatible with the
trainer's policy CE and the Rust priors contract.

### 3.5 Token budget vs full legal coverage (real tension)

Policy must cover **all** legal moves (≤1681 late game). If `max_candidates <
n_legal`, some moves get no logit. Plan: (1) **measure** the real legal-count
distribution from existing dense_cnn shards (`legal_off` diffs) before fixing the
budget — hex frontiers are sparse, so ~384–512 likely covers most positions;
(2) `overflow_policy="fail_fast"` during bring-up so truncation is loud; (3) add
a cheap shared **fallback logit** for the long tail only if the data demands it.
No silent caps.

### 3.6 Parameter budget

Match the running 96×8 baseline (~2.1M). Target `token_dim≈128`, `L_gnn≈2–3`,
`L_ctx≈2–4`, `L_cross≈2`, `L_self∈{0,1}`, 4 heads; verify with `sum(p.numel())`
and land within ~10% of the dense_cnn baseline. Report the exact count.

---

## 4. The hard problems, honestly

### 4.1 Inference throughput in the 512-sim MCTS loop — THE make-or-break risk

dense_cnn reaches ~58 pos/s at 512 sims with a fixed `(N,13,41,41)` CNN + TRT
FP16, after a zero-copy buffer war (the `PlaneBuffer` pyclass). A **message-
passing GNN is fundamentally harder**:

- **Variable graph size** (nodes/edges differ per position) resists static
  batching and a fixed export signature.
- **Explicit message passing** (`scatter`/`index_add_` over variable edge lists)
  **does not export cleanly to TensorRT** and is slow in eager torch.
- **The transformer half** is O(T²·d); fine if `T` is bounded and dense, costly
  if attention is sparse/dynamic.

**This does NOT mean abandoning the GNN.** It means the GNN must be *implemented*
in a form that batches and (ideally) exports. The throughput phase evaluates,
in order of preference:

1. **Fixed-shape padded GNN.** Pad node sets to fixed budgets (`max_stones`,
   `max_candidates`, `max_windows`) with key-padding masks; represent typed
   edges as **fixed-shape padded neighbor tensors** (gather a constant `K`
   neighbors per node per edge type) so message passing is dense, static-shape
   tensor ops — batchable and ONNX/TRT-exportable. This keeps the *explicit
   typed GNN* while making it exportable. **Preferred if it meets the gate.**
2. **Attention-bias realization (optimization option, NOT a replacement).** Fold
   the typed local message passing into **additive attention biases** on the
   context transformer (typed-edge bias + hex-distance/locality bias +
   adjacency bias). This is mathematically a typed graph aggregation expressed as
   masked dense attention — fully fixed-shape and TRT-friendly. We treat this as
   a *performance equivalent* of the GNN to fall back to **only if** the explicit
   message-passing GNN can't hit the throughput gate, and we validate that it
   preserves the GNN's behavior (accuracy parity check), so the model is still
   "a GNN" in effect. It is an optimization path, evaluated empirically — not a
   silent substitution.
3. **Torch-FP16 fallback (no TRT).** If neither (1) nor (2) exports but eager
   `scaled_dot_product_attention` / fused-gather throughput is workable, run
   self-play on torch FP16 and **normalize the comparison by search compute**
   (§9), accepting lower pos/s.

**Feasibility verdict:** a *truly dynamic* message-passing GNN is unlikely to
export to TRT and will probably run below dense_cnn's pos/s in eager torch. The
fixed-shape padded GNN (1) is the bet that keeps the explicit GNN *and* exports;
the attention-bias form (2) is the proven-exportable safety net; torch FP16 (3)
is the floor. **Phase 5 is the go/no-go gate** that picks the final form and
decides project viability. Be honest in the run notes about which form ships and
what it cost vs dense_cnn.

**Bake in from day one:** bounded, *measured* `T`; mirror the `PlaneBuffer`
zero-copy transport for a `(N,T,F)` f16 token payload + small int bias/edge
tables; cache context KV across the cross-attention blocks; profile with
`scripts/_profile_selfplay.py` before committing to TRT.

### 4.2 Variable-size graphs and batched MCTS eval

Solved by the **fixed node budgets + padding + key-padding masks** (and, for the
GNN, fixed-`K` padded neighbor tensors per edge type). Rust hands Python a
`(N,T,F)` buffer + edge/bias tables + per-row legal counts (CSR
`legal_row_offsets`); Python masks pads, runs one batched forward, writes
`priors_bytes` by reading the first `n_legal[row]` legal-token logits. No
per-position Python loop. **Variable-size collation in *training*** (batching
graphs of different sizes) is new work and is scoped in §5.

### 4.3 D6 symmetry on a graph/token representation

- Re-derive and re-test the hex-D6 group (`D6_SIZE=12`) and apply it **at
  read/expand time** via a per-(run,epoch) symmetry vector — exactly dense_cnn's
  discipline. D6 acts on each node's relative coordinate and on window axis
  labels; **node/edge identity is permutation-invariant**, so only coords/edge-
  geometry rotate.
- **Spill advantage:** a pure-token/graph rep anchored on relative coords is
  **not bound to the 41×41 square crop**, so dense_cnn's corner-spill problem
  largely disappears — the **full D6 group is usable with no identity fallback**.
  (Only if a local square-crop CNN feature is added back does spill return; the
  MVP avoids it.)
- **Equivariance test (non-negotiable):** applying D6 to the input and inverse-D6
  to the policy output must equal the un-augmented forward (within fp tolerance)
  for all 12 elements. This is the test that prevents subtly poisoning the model
  (the dense_cnn D6 lesson, `MEMORY.md`).

### 4.4 Capacity / fairness

Size to ~2.1M (§3.6). The honest fairness axis is **search compute and
wall-clock**, not params alone, because a GNN+transformer and a CNN have
different FLOPs/param and inference profiles. State all axes in the comparison
(§9).

---

## 5. Scope of the sample-generation + training rewrite (be explicit)

This is a **major rewrite**, not a reuse. What is emulated vs newly built:

**Emulate dense_cnn's discipline (mirror the patterns, adapt the code):**
- Raw-fact columnar NPZ shards with offset arrays + a `SCHEMA_VERSION`.
- Power-law replay window, md5 train/val split, batch-aligned output shards,
  JSON sidecars (`replay.py` mechanics).
- Per-epoch D6-at-read symmetry vector; checkpoint hygiene (no `sample_buffer`);
  config/plugin/test discipline.

**New work (largely rewritten for the graph rep):**
1. **Rust sample-gen (`sample_gen.rs`, from scratch):** emit typed-node facts +
   **typed-edge construction** (adjacency / line / candidate-of / recency /
   side-context) + tactical-window scoring. This is the bulk of the new Rust.
2. **Schema extension:** likely add graph-specific fields (edge lists per type,
   window facts) → a new `SCHEMA_VERSION`; or recompute edges/windows at expand
   time for the MVP (no field) and cache later if it's a measured bottleneck.
3. **Expand step (`expand_row_to_graph`, from scratch):** compact row + symmetry
   → typed node tensors `(T,F)` + typed padded neighbor/edge tensors + scattered
   `(1681,)` policy/opp targets + `(65)`-binned value/stvalue targets. Replaces
   dense_cnn's `expand_*_to_planes`.
4. **Variable-size graph collation/batching (new):** pad-to-budget + masks (and
   fixed-`K` neighbor tensors); deterministic, byte-stable, tested.
5. **New trainer (`HexgtTrainer`):** consumes the graph batch; **same loss
   weights, AMP, grad-clip, optimizer (AdamW), and reporting as dense_cnn** so
   the training discipline matches even though the data path differs.
6. **New inference module:** graph payload → `priors_bytes`/`values_bytes` per
   the §2.6 protocol.

Framing for the roadmap: **"emulate dense_cnn's discipline, but sample-gen and
training are largely rewritten for the graph representation."**

---

## 6. Rust changes

### 6.1 Sample generation (`sample_gen.rs`) — from scratch

- Author the typed-node raw facts and **typed edges** (§3.3). Tactical-window
  scoring can take the *idea* from the old hexformer candidate scoring
  (`tactical_radius`, `frontier_radius`) but is re-implemented and tested fresh.
- MVP: recompute edges/windows at expand time from `stones_qr`+`legal_ids` (no
  schema field, can read existing dense_cnn shards for the behavioral-clone
  bootstrap). Cache in the shard (schema bump) only if measured to be a
  bottleneck.

### 6.2 MCTS integration — model fresh, framework pattern from dense_cnn

- Build the new model's batched-PUCT MCTS by **following dense_cnn's Rust
  pattern** (`dense_cnn/rust/src/{mcts.rs,mcts_eval.rs,encoding.rs}`): a native
  session, a transposition cache, and the Python-callback eval loop. We do **not**
  fork hexformer's MCTS crate; we mirror dense_cnn's, retargeting the encoding to
  emit the **graph/token payload**.
- **Mirror dense_cnn's zero-copy `PlaneBuffer`** (`mcts_eval.rs:48-100`): a
  `#[pyclass]` exposing `__getbuffer__` over a `Vec<f16>` for the `(N,T,F)` token
  features; ship edge/bias tables and `legal_row_offsets` as small int `PyBytes`.
  Same payload-dict protocol (§2.6).
- Build wiring: new crate `#[path]`-included into `hexo_models/rust/src/lib.rs`,
  registered as `hexo_models._rust.hexgt`; rebuild the **`hexo_models`** package
  (`maturin develop -m packages\hexo_models\Cargo.toml --features python`).

---

## 7. Package layout, naming, and the deletion

**Create a new package `packages/hexo_models/hexgt/` from scratch, structured as
a sibling of `dense_cnn`** (mirror dense_cnn's module set), and **delete
`hexformer_ar`** outright. "Delete" concretely means:

- Remove `packages/hexo_models/hexformer_ar/` (Python + Rust).
- Remove its entry point from `packages/hexo_models/pyproject.toml` (and the
  source-include lists).
- Remove its `#[path]` include + `sys.modules` registration from
  `packages/hexo_models/rust/src/lib.rs:7-9,24-29`.
- Remove `configs/hexformer_ar.toml`.
- Drop any `hexformer_ar` test files.

(The deletion is part of the build work, not this planning pass.)

Layout (each module **authored fresh, modeled on the dense_cnn file of the same
name**):

```
packages/hexo_models/hexgt/
  python/hexo_models/hexgt/
    __init__.py        # stable public surface
    constants.py       # BOARD_SIZE=41, INPUT_CHANNELS=13, VALUE_BINS=65, token/edge budgets
    config.py          # parse_hexgt_config — modeled on dense_cnn/config.py
    coordinates.py     # axial/cube + pack_coord_id helpers, re-derived + tested
    d6.py              # hex-D6 group, re-derived + tested (full group, no crop fallback)
    architecture.py    # typed GNN + context transformer + cross-attn + dense_cnn-contract heads
    losses.py          # binned value (dense_cnn math) + policy/opp CE
    samples.py         # finalize (dense_cnn pattern) + expand_row_to_graph (new)
    replay.py          # replay window/shuffle/split, modeled on dense_cnn/replay.py
    compact_io.py      # columnar NPZ + graph expansion (new expansion target)
    inference.py       # graph payload → priors_bytes/values_bytes
    trainer.py         # HexgtTrainer: dense_cnn loss/AMP/clip discipline, graph batch
    checkpoints.py     # dense_cnn checkpoint shape; sample_buffer rejection; no buffer write
    selfplay.py        # modeled on dense_cnn/selfplay.py
    evaluation.py      # SealBot match loop, modeled on dense_cnn/evaluation.py
    player.py          # hexo_runner adapter, modeled on dense_cnn/player.py
    performance.py     # calibration
    trt_backend.py     # ONNX export of the fixed-shape model (Phase 5)
    plugin.py          # ModelPlugin, modeled on dense_cnn/plugin.py
  rust/src/
    lib.rs, constants.rs, state.rs           # modeled on dense_cnn rust
    mcts.rs, mcts_eval.rs, encoding.rs       # dense_cnn pattern, graph payload
    sample_gen.rs                            # new typed-node + typed-edge facts
```

Entry point (after deleting hexformer):
```toml
[project.entry-points."hexo_train.models"]
dense_cnn = "hexo_models.dense_cnn.plugin:get_plugin"
hexgt     = "hexo_models.hexgt.plugin:get_plugin"
```

---

## 8. Phased roadmap — from-scratch build modeled on dense_cnn

Each phase has a milestone gate. **No GPU contention with the live dense_cnn
run** — phases 0–4 are CPU-only; GPU phases (5+) wait until the 96×8 run frees
the GPU.

**Phase 0 — Delete + scaffold (CPU).** Delete `hexformer_ar` (§7). Create
`packages/hexo_models/hexgt/` mirroring dense_cnn's module set; `constants.py`,
`config.py`; stub `architecture.py` returning the exact dense_cnn output keys
with random weights; add the entry point.
*Gate:* package installs editable; `load_model_plugin` resolves `hexgt`;
`forward` returns `{"policy","value","opp_policy"[,"stvalue_*"]}` with shapes
`(N,1681)/(N,65)/(N,1681)`.

**Phase 1 — Contract-conformance tests (CPU).** Mirror dense_cnn's test files as
`tests/test_hexgt_*.py`: output keys/shapes; checkpoint round-trip +
`sample_buffer` rejection; config unknown-key rejection; policy scatter↔gather
equivalence.
*Gate:* all contract tests green on random weights.

**Phase 2 — GNN + transformer model body (CPU).** Implement the typed GNN
(message passing over typed edges, §3.1/§3.3), context transformer, legal-move
cross-attention, optional self-attention, and the dense_cnn-contract heads.
*Gate:* forward runs on a synthetic graph batch; D6 **equivariance test** passes
for all 12 elements (§4.3); overfits one tiny fixed batch (loss → ~0).

**Phase 3 — Sample-gen + expand rewrite (CPU). [MAJOR]** Author the Rust typed-
node + typed-edge sample-gen (§6.1) and `expand_row_to_graph` (§5). Validate the
scattered `(1681,)` policy target matches dense_cnn's expansion for shared rows
read from existing `runs/` shards (read-only). Implement variable-size graph
collation/batching.
*Gate:* byte-stable graph batches; policy/value targets match dense_cnn
expansion for shared rows; D6-at-read symmetry vector wired.

**Phase 4 — Trainer + MCTS integration (CPU/light). [MAJOR]** Implement
`HexgtTrainer` (dense_cnn loss/AMP/clip/optimizer discipline; §5.5). Implement
the Rust MCTS session (dense_cnn pattern) + zero-copy graph payload + Python
`inference.evaluate_payload` → `priors_bytes`/`values_bytes`; reuse dense_cnn's
priors validation.
*Gate:* a short CPU self-play run produces legal games end-to-end; priors
validation passes; transposition cache hits; a CPU training pass decreases loss.

**Phase 5 — Throughput + form decision (GPU). [MAKE-OR-BREAK GATE]** Profile the
explicit message-passing GNN's pos/s at 512 sims (`scripts/_profile_selfplay.py`).
Evaluate, in order: (1) fixed-shape padded GNN TRT export; (2) attention-bias
realization TRT export (validate accuracy parity vs the explicit GNN); (3) torch
FP16. Run the dense_cnn-style correctness gate.
*Gate (go/no-go):* pick the fastest form that meets the correctness gate and an
acceptable pos/s. If the explicit GNN exports/runs acceptably → ship it. If only
the attention-bias form does → ship it as the GNN's performant realization
(parity-checked). If only torch FP16 is workable → proceed, compute-normalize
(§9). If none is workable → stop and reconsider (shrink `T`/layers or rethink).

**Phase 6 — Bootstrap / cold-start (GPU, after dense_cnn frees GPU).**
(a) **Behavioral-clone** Model 2 supervised on existing dense_cnn shards (fast
signal it can fit the targets) → then (b) **cold-start RL** self-play from random
init (dense_cnn 64×4 path). Reuse the scratch-64 autonomy supervisor; watch
opening entropy (`forced_playout_k` / opening-temperature lessons, `MEMORY.md`).
*Gate:* training loss decreases on real targets; healthy self-play opening
diversity.

**Phase 7 — Head-to-head vs dense_cnn (GPU).** SealBot eval + direct matches
under matched search compute (§9).
*Gate:* Model 2 reaches a defined fraction of dense_cnn's SealBot win-rate at
matched compute — or a clear, honest verdict that it does not.

---

## 9. Head-to-head evaluation methodology

Reuse the *exact* dense_cnn eval harness (re-implemented in `hexgt`, modeled on
dense_cnn) so the comparison is apples-to-apples:

- **Same SealBot eval:** `evaluate_epoch` → `hexo_runner.modes.match.run_match`
  vs `SealBotPlayer` at `sealbot_variant="best"`, `time_limit=0.05`, alternating
  colors, `games_per_epoch=64`. Eval uses `use_trt=False` for both (TRT-
  independent strength).
- **Same MCTS:** same batched PUCT + transposition cache + same `search_visits`
  (512 for strength). Only the network + payload encoding differ.
- **Matched compute — lead with this.** Report three axes: (1) **matched search
  visits** (per-search quality, the "is the network smarter" signal); (2)
  **matched wall-clock self-play budget** (penalizes slow inference honestly);
  (3) **matched param count** (~2.1M, the capacity footnote).
- **Direct match:** Model 2 vs dense_cnn (`run_match`, alternating colors, ≥200
  games), win-rate ± Wilson interval.
- **Same opening-diversity controls** (`opening_moves`/`opening_temperature`,
  per-(game,move) seeds — the eval-diversity lesson in `MEMORY.md`).

---

## 10. Decisions reflected + remaining open questions

**Binding decisions reflected in this revision:**
1. **Delete `hexformer_ar`; build fresh modeled on dense_cnn** (§0, §1.2, §7,
   §8 Phase 0). hexformer's code/scaffolding is not the foundation; dense_cnn's
   patterns are.
2. **The typed GNN is first-class** (§3.1, §3.3). The attention-bias / fixed-
   shape formulation is an **optimization option evaluated in Phase 5**, not a
   silent replacement (§4.1).
3. **Major sample-gen + training rewrite, stated explicitly** (§5, §6, §8 Phases
   3–4). We emulate dense_cnn's *discipline*; the graph sample-gen, expansion,
   collation, and trainer are largely new.

**Retained from the prior analysis (still in force):** the Phase-5 throughput
gate, drop-in contract matching for a fair comparison (§2, §3.4), full-group D6
with the equivariance test (§4.3), and the matched-compute fairness methodology
(§9).

**Open questions for the user:**
1. **Package name:** `hexgt` (proposed) acceptable?
2. **Auxiliary heads:** confirm dropping WDL/threat/lookahead from the MVP
   contract (re-add later as private aux) is acceptable.
3. **Token budget fallback:** OK to start `fail_fast` and add a long-tail
   fallback logit only if measured legal counts demand it?
4. **Bootstrap:** behavioral-clone from dense_cnn shards first (recommended), or
   straight to cold-start RL for a "clean" comparison?
5. **If Phase 5 shows the explicit message-passing GNN can't hit the throughput
   gate**, is the attention-bias realization (parity-checked, still a GNN in
   effect) an acceptable ship form, or must the explicit message-passing form be
   preserved even at a throughput cost?

---

## Appendix A — file:line index of contracts referenced

- dense_cnn forward: `packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/architecture.py:202-220`
- dense_cnn value bins: `.../dense_cnn/losses.py:20-30`
- dense_cnn constants/planes: `.../dense_cnn/constants.py:9-28`
- dense_cnn geometry/flat index: `.../dense_cnn/geometry.py:36-62`
- dense_cnn ActionId pack: `.../dense_cnn/d6.py:19,41-59`
- dense_cnn compact schema: `.../dense_cnn/compact_io.py:33,49-216,243-322`
- dense_cnn replay window/shuffle: `.../dense_cnn/replay.py:523-535,615-755`
- dense_cnn checkpoint: `.../dense_cnn/checkpoints.py:23-105`
- dense_cnn config: `.../dense_cnn/config.py:183-318`
- dense_cnn plugin/entry point: `.../dense_cnn/plugin.py:27-119`, `packages/hexo_models/pyproject.toml:17-19`
- Rust↔Python evaluator payload: `.../dense_cnn/rust/src/mcts_eval.rs:48-100,315-390`
- TRT: `.../dense_cnn/trt_backend.py:93-396`
- ModelPlugin protocol: `packages/hexo_train/python/hexo_train/registry.py:24-103`
- hexformer (to delete) architecture: `.../hexformer_ar/architecture.py:15-265`
- hexformer (to delete) Rust include: `packages/hexo_models/rust/src/lib.rs:7-9,24-29`
