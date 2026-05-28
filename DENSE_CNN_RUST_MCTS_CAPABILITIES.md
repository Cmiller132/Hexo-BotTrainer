# Dense CNN Rust MCTS Capabilities

The native dense_cnn MCTS lives in `packages/hexo_models/dense_cnn/rust/src`
(`mcts.rs`, `mcts_tree.rs`, `mcts_eval.rs`). It is a batched PUCT searcher whose
neural evaluation is delegated to a Python/Torch callback. `capabilities()`
(`lib.rs`) advertises the native paths that exist; the authoritative behavioral
description is `packages/hexo_models/dense_cnn/README.md`.

## Capability flags

`hexo_models._rust.dense_cnn.capabilities()` reports:

- `model_family = "dense_cnn"`, `state_source = "direct_engine_state"`,
  `coordinate_encoding = "u32_i16_pair"`
- `model1_batch_inputs` — native f32 plane encoding for direct inference.
- `model1_mcts_session_search` / `model1_mcts_tree_reuse_session` — a stateful
  session that promotes the selected child subtree after each move
  (`KataGo_Search_makeMove_promote_child`).
- `model1_mcts_all_legal_candidates` — every in-crop legal move is a search
  candidate; there is no progressive widening, candidate cap, or hidden-prior
  mass.
- `model1_mcts_root_dirichlet_noise` — total-alpha Dirichlet noise at the root
  (per-action `alpha = total_alpha / legal_count`).
- `model1_mcts_root_policy_temperature` — root policy softmax temperature applied
  before Dirichlet noise.
- `model1_mcts_first_play_urgency`, `model1_mcts_virtual_loss` — FPU reduction
  for unvisited edges and virtual loss for batched leaf gathering.
- `model1_sample_from_state` — state-derived compact sample facts.

## Search structure

- **Nodes** store statistics and outgoing edges only; leaf states are recreated
  by replaying edge actions from a root clone, so `hexo_engine` stays the single
  source of move legality and terminal detection.
- **Lazy-staged edges:** a node holds its legal priors as `unexpanded_priors` and
  materializes an edge only when PUCT selects that candidate over the best
  existing edge.
- **PUCT:** `score = value_or_fpu(parent_value, fpu_reduction) + prior * c_puct *
  sqrt(parent_visits) / (1 + edge_visits)`.
- **Virtual loss:** reserves a path during batched leaf gathering and is added
  back with the real (perspective-negated) leaf value on backup.
- **Tree reuse:** after a move the chosen child subtree is promoted to the new
  root; a root whose state hash no longer matches the live engine state is
  discarded and re-evaluated.

## Evaluator boundary (single mode)

Rust batches unique uncached leaf states, encodes f32 planes, and sends one
payload to `DenseCNNInference.evaluate_model1_payload`:

- Request: `inputs` (f32 NCHW bytes), `shape`, `legal_flat_indices_bytes` (the
  per-row in-crop legal crop flats), `legal_row_offsets` (CSR offsets).
- Response: `values_bytes` (one f32 per row, range-checked to `[-1, 1]`) and
  `priors_bytes` (one positional f32 per legal flat).

Both sides reject malformed payloads (wrong byte lengths, non-finite values,
duplicate actions, zero prior mass). An exact-hash evaluation cache coalesces
transpositions and duplicate leaf requests.

## What was removed

The earlier implementation also carried progressive widening (Chaslot-style
unpruning), a candidate-limited half-precision evaluator mode with on-device
top-k prior selection, hidden-prior mass for omitted in-crop actions, lazy legal
materialization, and tactical win/block prior injection. Those were removed in
favor of KataGo's simpler "search all legal candidates" approach; the evaluator
now has a single full-prior mode.
