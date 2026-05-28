# Dense CNN Model 1 Code Guide

This directory is the complete implementation of the production dense CNN
model family. It owns model-specific acceleration, tensor encoding, search,
self-play sample generation, training targets, checkpoint integration, and the
training plugin entry point.

The code is intentionally split into a Python half and a Rust half:

- Python owns PyTorch, configuration, plugin wiring, training loops, checkpoint
  IO, and small wrapper objects that make the native outputs convenient to use.
- Rust owns live `hexo_engine.HexoState` intake, model-specific tensor encoding
  from engine states, batched MCTS tree search, strict evaluator byte-payload
  parsing, and compact sample generation/finalization from authoritative game
  states.

The production path is one explicit path:

1. `hexo_train` loads `DenseCNNPlugin` from `python/.../plugin.py`.
2. The plugin parses `configs/dense_cnn_model1.toml` through
   `python/.../config.py`.
3. The plugin builds `Model1Network`, `SampleBuffer`, `DenseCNNTrainer`, the
   checkpoint loader/saver, self-play callback, evaluation callback, and sample
   finalizer.
4. Self-play creates live `hexo_engine.HexoState` games.
5. `python/.../mcts.py` calls the dense-cnn Rust MCTS session.
6. Rust clones live engine states through `rust/src/state.rs`.
7. Rust MCTS encodes leaf states and calls
   `DenseCNNInference.evaluate_model1_payload`.
8. Python/Torch returns strict byte payloads: `values_bytes`, `priors_bytes`,
   and candidate metadata when progressive widening asks for top-k priors.
9. Rust finalizes search results into compact byte-backed visit policies.
10. Python records samples from live states through Rust sample generation.
11. Rust finalizes value, opponent-policy, and lookahead targets at game end.
12. Python compresses samples, stores them in `SampleBuffer`, expands sampled
    records into tensors, computes losses, and trains the network.

There is no production fallback to legacy payload formats, Python mirror state,
raw mapping replay entries, one-shot MCTS wrappers, or silent repair of invalid
inputs. Invalid config, evaluator payloads, native search arguments, and sample
schema mismatches should fail at the boundary that receives them.

## Directory Layout

`python/hexo_models/dense_cnn/` contains the Python model package:

| File | Role |
| --- | --- |
| `__init__.py` | Public Python export surface for architecture, config, losses, inference, and calibration. |
| `architecture.py` | PyTorch Model 1 network: hex-masked convolutions, residual trunk, policy head, value head, opponent-policy head, lookahead heads, and inference-time conv/batchnorm fusion. |
| `checkpoints.py` | Model, optimizer, trainer, and replay-buffer checkpoint loading/saving for the generic training pipeline. |
| `config.py` | Strict TOML-facing config dataclasses and validation. Unknown keys and invalid scalar values raise during parsing. |
| `constants.py` | Python copy of model tensor dimensions and input-plane indices. Must stay aligned with Rust `constants.rs`. |
| `d6.py` | Axial coordinate D6 symmetry transforms used by sample augmentation and dense target projection. |
| `debug_artifacts.py` | Optional bounded PNG renderer for inspection tooling. It is not part of the self-play production path. |
| `evaluation.py` | Epoch evaluation against SealBot through the generic runner. |
| `geometry.py` | Crop-centered axial coordinate helpers shared by Python input and target projection. |
| `inference.py` | PyTorch inference adapter and strict Rust MCTS evaluator callback. This is the only Python/Torch evaluation boundary used by native MCTS. |
| `input.py` | Expansion of compact sample facts into dense input planes, policy targets, and legal masks. |
| `losses.py` | Policy, value-bin, opponent-policy, lookahead, and combined Model 1 loss functions. |
| `mcts.py` | Python wrapper around the required Rust MCTS session. It exposes `BatchedMctsSession` and byte-backed `CompactVisitPolicy`. |
| `performance.py` | Calibration probes for inference batch size, self-play batch settings, training batch size, and diagnostic throughput reporting. |
| `player.py` | Generic runner `Player` adapter that uses dense CNN inference plus the persistent Rust MCTS session. |
| `plugin.py` | Training plugin entry point consumed by `hexo_train`. It wires all dense-cnn components into generic training. |
| `rust_bridge.py` | Thin import and call boundary for `hexo_models._rust.dense_cnn`. It deliberately delegates native argument validation to Rust. |
| `samples.py` | Compact sample dataclasses, compressed replay buffer, Rust sample bridge, D6 expansion, tensor stacking, and checkpoint schema validation. |
| `samples_finalizer.py` | Generic training-pipeline finalizer adapter. Dense-cnn self-play finalizes samples immediately, so this reports buffer state. |
| `selfplay.py` | Sequential active-game self-play loop that uses live engine states, persistent MCTS sessions, Rust sample generation, and Rust finalization. |
| `trainer.py` | Optimizer-backed training over compressed samples expanded with pipeline-provided D6 symmetries. |

`rust/src/` contains the native dense-cnn extension modules:

| File | Role |
| --- | --- |
| `lib.rs` | Registers the dense-cnn PyO3 functions/classes into `hexo_models._rust.dense_cnn` and publishes capability metadata. |
| `constants.rs` | Rust copy of model dimensions, plane indices, chunk sizes, and base-plane caches. Must stay aligned with Python constants. |
| `state.rs` | Live `hexo_engine.HexoState` intake through the engine capsule API. This keeps the engine generic while letting dense_cnn clone authoritative states. |
| `encoding.rs` | Native Model 1 tensor encoding from `HexoState`, including crop selection, legal planes, recency planes, hot-cell planes, and optional half-precision MCTS payloads. |
| `mcts_eval.rs` | Batched evaluator adapter, exact evaluation cache, strict byte-payload parser, prior validation, and cache diagnostics. |
| `mcts_tree.rs` | PUCT search tree, progressive widening, hidden-prior staging, tactical candidates, root promotion, virtual visits, and backups. |
| `mcts.rs` | PyO3 `Model1MctsSession` boundary: session state, argument validation, batched root search, result payloads, and visit-policy sampling. |
| `sample_gen.rs` | Native compact sample generation from live states and game-end target finalization. |

## Python Flow

The Python dense_cnn package has four main execution paths.

### Training Plugin Construction

`plugin.py` is loaded by the generic `hexo_train` registry. Its work is mostly
composition:

1. Parse model config with `parse_model1_config`.
2. Construct `Model1Network` using architecture config.
3. Construct `SampleBuffer` using sample config.
4. Construct the optimizer from training config.
5. Create `DenseCNNTrainer`.
6. Return dense-cnn overrides for checkpointing, performance calibration,
   self-play, sample finalization, training, and evaluation.

The generic training package does not know about dense input planes, MCTS
payloads, or Model 1 heads. It only calls the plugin-provided components.

### Self-Play

`selfplay.py` owns the production sample-generation loop. The important objects
are live engine states, not serialized state mirrors.

1. Build a `DenseCNNInference` wrapper around the current PyTorch model.
2. Create a native `BatchedMctsSession`.
3. Start up to the configured number of active games with `engine.new_game`.
4. Choose playable games that have not ended and are below `max_actions`.
5. Search only enough positions to satisfy the sample budget.
6. Send each searched live state to `BatchedMctsSession.run`.
7. Use the returned visit policy and root value to create a pre-decision sample
   through `samples.sample_from_state`.
8. `sample_from_state` calls Rust, which clones the live engine state and emits
   compact facts such as stones, legal action ids, placement history, hot cells,
   current player, phase, and crop center.
9. Compress each pending sample and store it with the player label and root
   value until the game finishes.
10. Apply the selected action to the live engine state.
11. If the sample budget is met while active games remain unfinished, roll out
    the rest using direct dense-cnn policy inference instead of MCTS, so every
    pending MCTS sample can still receive a final outcome.
12. At game end, write an `.hxr` record and call `finalize_game_samples`.
13. `finalize_game_samples` calls Rust to set final value targets, opponent
    policy targets, lookahead targets, and schema metadata.
14. Add finalized compressed samples to `SampleBuffer`.
15. Discard the game key from the native MCTS session so stale subtrees are not
    reused for a finished game.

The self-play loop keeps the strict boundary rule: invalid native settings,
invalid model outputs, impossible game state, or malformed sample payloads raise
instead of being repaired locally.

### Inference And MCTS Evaluation

`inference.py` is used in two ways:

- Direct state inference for player policy rollouts and evaluation.
- Native MCTS callback evaluation through `evaluate_model1_payload`.

Direct state inference calls `rust_bridge.model1_batch_inputs`, which delegates
encoding to Rust and returns:

- `inputs`: contiguous float32 bytes.
- `shape`: `(batch, channels, board_size, board_size)`.
- `legal_action_ids`: per-row packed coordinates.
- `legal_flat_indices`: per-row dense crop indices.
- `centers`: per-row crop centers.

`DenseCNNInference` converts the input bytes to a tensor, runs the network, and
returns `InferenceResult` objects with legal priors projected from logits.

Native MCTS evaluation uses a stricter byte protocol. Rust sends:

- `inputs`: contiguous float32 or float16 tensor bytes.
- `input_dtype`: optional, defaults to `float32`.
- `shape`: exact 4D tensor shape.
- Either explicit legal index bytes plus row offsets, or `legal_mask_from_inputs`
  with `max_prior_candidates` for candidate-limited MCTS.

Python returns:

- `values_bytes`: float32 scalar values in row order.
- `priors_bytes`: float32 priors matching either legal row offsets or selected
  top-k candidates.
- `selected_flat_indices_bytes` and `selected_row_offsets` when Rust requested
  candidate-limited priors from the legal plane.

The callback validates shape and byte lengths before tensor views are created,
because wrong byte sizes otherwise become confusing tensor reshape failures or,
worse, incorrect priors.

### Training

`trainer.py` trains from compressed samples:

1. `select_training_samples` asks `SampleBuffer` for a recency-weighted sample
   window.
2. The generic training pipeline supplies one D6 symmetry per selected sample.
3. `train_passes` decodes each compressed sample, expands it with the assigned
   symmetry, fills missing lookahead targets with zero-valued masked tensors,
   stacks tensors by key, and moves them to the selected device.
4. `Model1Network` returns policy, value, opponent-policy, and lookahead heads.
5. `losses.model1_loss` computes the weighted combined loss.
6. AMP and gradient clipping are applied according to config.
7. Optimizer steps are counted explicitly; a pass that performs no steps raises.

`input.py` and `losses.py` are the strict target side of this path. They project
policy weights into the crop, reject negative or non-finite target weights, and
reject value targets outside `[-1, 1]`.

## Rust Flow

The Rust side exists so model-specific search can work directly from the engine
without making `hexo_engine` know about dense-cnn tensors or MCTS details.

### State Intake

`state.rs` imports `hexo_engine._rust.state_api_capsule()` and checks the
capsule version. Each Python `HexoState` is cloned into a Rust `HexoState`, then
the temporary capsule handle is released. The dense-cnn code works on that clone.

This boundary is deliberately narrow:

- The engine remains generic.
- Dense-cnn gets authoritative state, legal moves, history, phase, and outcome.
- Python never sends move-history payloads to reconstruct state inside dense_cnn.

### Tensor Encoding

`encoding.rs` converts `HexoState` into the 13-plane Model 1 tensor. The planes
are:

1. Current-player stones.
2. Opponent stones.
3. Empty cells.
4. Legal cells.
5. Second-placement phase marker.
6. First stone of a two-stone turn.
7. Current-player color marker.
8. Current-player recency.
9. Opponent recency.
10. Opponent hot cells.
11. Current-player hot cells.
12. Distance from crop center.
13. Opponent last-turn placements.

The crop center is the rounded mean of occupied axial coordinates, or `(0, 0)`
for an empty board. Coordinates outside the crop are omitted from the dense
view. MCTS can request full float32 inputs with explicit legal indices, or
float16 inputs with the legal plane included so Python can choose top-k legal
candidate priors without shipping every legal index twice.

### MCTS Session

`mcts.rs` exposes `Model1MctsSession`. The session stores:

- A search tree per active game key.
- One shared evaluation cache.
- The configured evaluation-cache capacity.

For each `search` call:

1. Validate native scalar arguments.
2. Clone live Python states through `state.rs`.
3. Ensure game keys and states have the same length.
4. Reuse a previous tree only if the stored root hash matches the incoming live
   state.
5. Evaluate missing roots through `mcts_eval.rs`.
6. Create or reuse `RustSearch` objects from `mcts_tree.rs`.
7. Record root visit baselines so the returned policy can represent only visits
   added by this call.
8. Run root searches to the requested visit count.
9. Build diagnostics for tree and evaluator behavior.
10. Sample one selected action from the visit policy.
11. Return a byte-backed result payload to Python.
12. Promote each selected child as the next root if it exists and is nonterminal.

The session is the reason self-play can reuse search work across turns without
keeping model-specific logic in `hexo_engine`.

### Evaluation Adapter

`mcts_eval.rs` owns the Rust-to-Python neural evaluator contract.

1. Hash each state with `hexo_utils::hash_state`.
2. Resolve cache hits and duplicate states before calling Python.
3. Encode unique states into contiguous input bytes.
4. Build the exact payload expected by `DenseCNNInference.evaluate_model1_payload`.
5. Call Python once per chunk.
6. Require exact byte lengths for returned values and priors.
7. Convert selected crop flats back into packed engine coordinates when using
   candidate-limited priors.
8. Verify priors are finite, nonnegative, unique per row, legal, and nonempty for
   nonterminal rows.
9. Insert exact evaluations into the bounded cache.

The cache is cleared if the prior-candidate mode changes, because full-prior and
candidate-limited evaluations are not interchangeable.

### Tree Mechanics

`mcts_tree.rs` is the search engine. A `RustSearch` owns one root state, an arena
of nodes, and a state-hash table. Nodes store values, visit counts, active edges,
and hidden prior candidates. They do not store a full state at every node.

Selection recreates the leaf state by cloning the root state and replaying the
selected edge actions. That keeps memory use low and guarantees game rules still
come from `hexo_engine`.

The tree uses:

- PUCT edge scoring with first-play urgency.
- Progressive widening for large legal action sets.
- Hidden prior mass for legal actions not yet materialized as edges.
- Tactical candidate protection for immediate wins and opponent blocks.
- Root Dirichlet noise for self-play exploration.
- Virtual visits so batched leaf selection does not select the same unevaluated
  edge repeatedly.
- Root promotion so the selected child subtree becomes the next turn's root.

### Sample Generation

`sample_gen.rs` creates compact samples from live engine states and finalizes
targets after a game.

`model1_sample_from_state` reads current facts:

- Game id and turn index.
- Current player and phase.
- Crop center.
- Stones.
- Legal in-crop action ids.
- Full placement history.
- First stone for second-placement turns.
- Hot cells.
- Opponent last-turn cells.
- MCTS policy weights.
- Current root value.
- Optional opponent policy, lookahead, and metadata.

`model1_finalize_game_samples` receives pending decisions for one game and:

- Sets final value target from winner relative to each sample's player.
- Finds the next future opponent MCTS policy for the opponent-policy target.
- Computes lookahead value targets from future root values at requested horizons.
- Marks truncation metadata for max-action draws.
- Writes the current target schema version and lookahead semantics.

Python then validates and compresses the finalized sample through `samples.py`.

## Boundary Rules

The dense_cnn code keeps checks at true boundaries:

- Config parsing rejects unknown keys and invalid values.
- Rust session entry rejects invalid search settings and mismatched state/key
  batches.
- Python evaluator callback rejects malformed byte payloads before tensor views.
- Rust evaluator parser rejects missing bytes, wrong lengths, illegal selected
  candidates, duplicate priors, invalid values, and zero prior mass.
- Sample-buffer loading rejects non-current replay schema and malformed compressed
  samples.
- Loss functions reject invalid targets before normalization.

The thin Python wrapper in `mcts.py` and `rust_bridge.py` intentionally avoids
duplicating native scalar validation. Those wrappers are transport surfaces; the
native Rust session is the search boundary.

## Keeping The Python And Rust Halves Aligned

When changing Model 1 representation, update both halves together:

- Plane indices: `python/.../constants.py` and `rust/src/constants.rs`.
- Crop projection: `python/.../geometry.py`, `python/.../input.py`, and
  `rust/src/encoding.rs`.
- Sample schema: `python/.../samples.py` and `rust/src/sample_gen.rs`.
- MCTS evaluator payload: `python/.../inference.py` and `rust/src/mcts_eval.rs`.
- Public exports: `python/.../__init__.py`, `python/.../plugin.py`, and
  `rust/src/lib.rs`.

Focused tests live under `tests/test_dense_cnn_*.py`. Rust compile and unit
tests require the local Rust toolchain; if `cargo` is unavailable, Python tests
can still validate Python syntax and the mocked Python/Rust boundary contracts,
but native compilation remains unverified.
