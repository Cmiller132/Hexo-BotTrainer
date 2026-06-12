# hexo_models.dense_cnn ("Model 1")

The original dense-CNN AlphaZero-style lineage for Hexo: a 13-plane 41x41
hex-disk crop encoder, a gated-residual CNN with a 65-bin value head, a
KataGo-style compact-sample/NPZ replay pipeline, and a Rust/PyO3 batched PUCT
MCTS.

## Status (read this first)

The package has a split active/legacy status:

| Half | Status | Why |
|---|---|---|
| Python (`python/hexo_models/dense_cnn/`) | **Legacy** | Superseded by `packages/dense_cnn_restnet`, a full Python fork (ResTNet trunk). Kept loadable for old checkpoints, the dashboard debug worker, ~14 test files, and `hexgnn`'s `compact_io` dependency. |
| Rust (`rust/src/`) | **Active production** | `dense_cnn_restnet` ships no Rust of its own. The live `main_4` run (and `main_1`..`main_3` before it) drives this crate's encoder, MCTS sessions (including the restnet-only `run_continuous` scheduler), and sample-fact builder through `hexo_models._rust.dense_cnn`. |

Runs that used this lineage directly (all retired): `configs/dense_cnn_model1*.toml`
(Windows-native model1 era) and `configs/dense_cnn_rl_main1.toml`. The active
restnet runs (`configs/dense_cnn_restnet_main_*.toml`) use only the Rust half.

Consequence: **rebuilding the crate for restnet changes this lineage's search
semantics too** (single shared `hexo_models._rust` module). Fixes to the Python
half must usually be mirrored by hand into the restnet fork, and vice versa.

## Architecture (`python/.../architecture.py`)

`Model1Network`:

| Piece | Detail |
|---|---|
| Input | `(N, 13, 41, 41)` float32 crop tensor |
| Stem | `HexConv2d` 13 -> C, ReLU |
| Trunk | `blocks` x `GatedResBlock` (residual + sigmoid-gated main branch); defaults C=96, blocks=6 (`constants.DEFAULT_CHANNELS/_BLOCKS`) |
| `policy` head | `PolicyHead`: fully-convolutional, one logit per crop cell, flattened to `(N, 1681)` ("P7" fix — replaced the old 5.6M-param FC head that echoed diffuse priors) |
| `value` head | `ValueBinnedHead`: KataGo-style 65-bin distribution over `[-1, 1]` |
| `opp_policy` head | Second `PolicyHead`; target = next opponent MCTS policy |
| `stvalue_<h>` heads | Optional `ValueBinnedHead` per configured short-term-value horizon (EMA of future root values) |

`HexConv2d` is a normal 3x3 conv with kernel corners `(0,0)` and `(2,2)`
masked to zero, so the square-grid receptive field matches the six axial hex
neighbors. `forward_policy_value()` skips the aux heads for search batches.
`optimized_model1_for_inference()` returns an eval clone with hex masks baked
into plain `nn.Conv2d` and conv+BN fused, used for CUDA inference.

## Encoding / features

The board is projected into a fixed 41x41 square crop covering a **radius-20
hex disk** around a crop center; cells outside the disk are dead. The
contract exists in two languages and must be kept in sync by hand:

- Rust (production path): `rust/src/encoding.rs` + `rust/src/constants.rs` —
  encodes live `HexoState` objects for inference and MCTS leaves.
- Python (training/expand path): `constants.py`, `geometry.py`, `input.py` —
  re-expands stored compact facts into identical tensors.

The 13 planes (`constants.py`):

| Idx | Plane | Idx | Plane |
|---|---|---|---|
| 0 | own stones | 7 | own recency |
| 1 | opponent stones | 8 | opponent recency |
| 2 | empty | 9 | opponent hot (threat) cells |
| 3 | legal | 10 | own hot cells |
| 4 | second placement of turn | 11 | center distance |
| 5 | first stone of turn | 12 | opponent last turn |
| 6 | player colour | | |

Known design limitation: the encoder **intentionally excludes legal engine
moves outside the radius-20 crop** from the policy and from MCTS
(`encoding.rs`, `Model1EncodedState.all_legal_action_count` comment). This
freeze-out of out-of-rim wins was the root cause of the restnet `main_3`
collapse (see `docs/analysis/MAIN4_RECOMMENDATION.md`); every consumer of this
Rust inherits it.

## Sample and shard formats

Self-play stores compact *facts*, not tensors; tensors are rebuilt at train
time so D6 symmetry can be re-randomized per epoch.

| Layer | File | Format |
|---|---|---|
| Per-position sample | `samples.py` (`Model1SampleData`, target schema v4) | Packed stones/history (int16 coords + 1-byte owners), compact visit policy, value/opp-policy/STV targets. Facts built by Rust `sample_gen.rs`; targets attached in Python by `finalize_game_samples` (z from winner, opp policy from next opponent decision, STV = EMA of future root values). |
| Compact shard | `compact_io.py` (`COMPACT_SCHEMA_VERSION = 1`) | One columnar `.npz` per game: fixed per-row scalar arrays plus, for each variable-length field, a concatenated data array + `int64` offsets array of length N+1. Root prior, `policy_surprise`, `frequency_weight` are dropped at write (surprise weighting is pre-baked as row duplication by `replay.materialize_policy_surprise_rows`). |
| Expanded training rows | `replay.py` `NPZ_KEYS` | `inputNCHW`, `policyTargetsNCHW`, `oppPolicyTargetsNCHW`, `rootPolicyNCHW`, `legalMaskNCHW`, `valueTargetsN`, `shortTermValueTargetsNC`, `shortTermValueMasksNC`, `metadataInputNC` — produced by the KataGo-style shuffle (`build_katago_shuffle`). |

The compact shard format is a cross-lineage contract: `packages/hexgnn`
imports `compact_io` directly, and the restnet fork's copy must stay
byte-compatible.

## Training approach

- **Plugin**: `plugin.py` registers as `dense_cnn` in the `hexo_train.models`
  entry-point group; the generic pipeline (`python -m hexo_train.cli.train_model`)
  calls `build_model` / `training_component_overrides` / `generate_selfplay` /
  `evaluate_epoch` / `calibrate_performance` per epoch.
- **Self-play** (`selfplay.py`): batches all active games through one
  persistent Rust MCTS session (the epoch-batched `run` API; the `run_continuous`
  per-slot scheduler in the same Rust is used only by restnet). Full search
  over all legal in-crop moves, no rollouts, no progressive widening; writes
  `.hxr` game records plus compact `.npz` shards and live-progress JSON
  (`dense_cnn.selfplay.live.json`) for the dashboard.
- **Replay/training** (`replay.py`, `trainer.py`): mtime-ordered shard window,
  KataGo-style shuffle with train/validation md5 split; `DenseCNNTrainer`
  expands rows at read time under a fresh per-epoch D6 symmetry via a
  process pool, then runs AMP optimizer steps with `losses.model1_loss`
  (policy CE + 65-bin value CE + opp-policy + optional STV terms).
- **Evaluation** (`evaluation.py`): per-epoch games vs the external SealBot
  minimax baseline through `hexo_runner`.
- **Checkpoints** (`checkpoints.py`): `.pt` payload
  `{model: "hexo_models.dense_cnn", model_state, optimizer_state, train_state, epoch}`
  with optional `.txt` pointer indirection; strict shape checks on resume.
- **Calibration** (`performance.py`): measured probes pick inference/optimizer/
  self-play batch sizes before the first epoch.

## Rust side (`rust/src/`)

Not a standalone crate — `#[path]`-included by
`packages/hexo_models/rust/src/lib.rs` and exposed as the
`hexo_models._rust.dense_cnn` submodule (rebuild via
`scripts/_rebuild_hexo_models_hexgt.sh`, maturin, WSL venv).

| File | Role |
|---|---|
| `encoding.rs` | `HexoState` -> 13-plane f32 tensor + legal crop-flat rows; `model1_batch_inputs` pyfunction |
| `mcts.rs` | `Model1MctsSession`: per-game tree reuse; `run` (epoch-batched) and `run_continuous` (per-slot scheduler with PCR/policy-init, restnet-only consumer) |
| `mcts_tree.rs` | PUCT mechanics: lazy candidates, Dirichlet noise, FPU, forced playouts, virtual loss, root promotion |
| `mcts_eval.rs` | Evaluator boundary: state hash/dedup/cache, strict byte protocol to Python (`evaluate_model1_payload` returns exact-length `values_bytes`/`priors_bytes`) |
| `sample_gen.rs` | `model1_sample_from_state`: compact per-position facts for self-play |
| `state.rs` | Clones live Python `HexoState` via the `hexo_engine._rust` state-API capsule (v2) |
| `constants.rs` | Rust mirror of the tensor contract (must match `constants.py`) |

TSS (threat-space search) semantics come from the shared
`packages/hexo_models/rust/src/threats_shared.rs` (one definition shared with
the hexgt lineage).

## Connections

| Consumer | What it uses |
|---|---|
| `packages/dense_cnn_restnet` (ACTIVE) | The Rust module read-only via its forked `rust_bridge.py`; near-identical Python fork of most modules here |
| `packages/hexgnn` (parked) | `compact_io.write/read_compact_shard`, `samples`, `replay.materialize_policy_surprise_rows` |
| `packages/hexo_frontend` | `debug_infer.py` loads `hexo_models.dense_cnn`-tagged checkpoints for the dashboard debug screen and Arena bots |
| `packages/hexo_engine` | Game truth (Python API) + Rust rlib + state capsule |
| `packages/hexo_runner` | Player protocol, `.hxr` records, SealBot eval opponent |
| `packages/hexo_train` | Plugin discovery and the epoch loop |

## Tests

`tests/test_dense_cnn_*.py` (~14 files: pipeline, compact_io, replay schema,
sample generation, TSS, inference bucketing, pool lifecycle, temperature
schedule) plus `tests/test_hexo_models_architecture.py` /
`test_hexo_models_samples.py`. Authoritative only in the WSL `hexgt-build`
venv (the native `.so` is Linux-only). Note: `test_dense_cnn_compact_io.py`
has 4 known-stale failures asserting pre-disk-crop semantics.
