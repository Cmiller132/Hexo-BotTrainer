# hexgnn

## Purpose and status

`hexgnn` is the "Model 2" GNN lineage: a deliberately stripped-down fork of the
`hexo_models/hexgt` lineage. It keeps the relational typed-GNN message-passing
trunk but drops the context transformer and the short-term-value (STV) heads;
the heads are policy + 65-bin value + opponent policy. The package is a
complete RL vertical: config, D6-invariant dynamic-graph featurizer,
packed-graph collation, trainer, self-play (PCR / soft-Z / policy-surprise),
SealBot evaluation, a `hexo_runner` player adapter, and a forked Rust
accelerator (candidates / featurizer / MCTS / TSS threats).

**Status: PARKED / legacy.** Per `HANDOFF.md` ("hexgnn was explored and set
aside (not the active path)"; "GNN experiment (parked)"). The active training
lineage is `packages/dense_cnn_restnet`. hexgnn remains buildable and
referenced (driver scripts, `configs/hexgnn_model.toml`, 9 test files), and
its Rust crate is compiled into every `hexo_models` native build.

## Design lineage

hexgnn was forked from hexgt so the sparse-graph featurizer and performance
rewrite could evolve independently of the (halted) hexgt run. The
evaluation / inference / mcts / player modules and the whole Rust tree are
forks of their `hexo_models/hexgt` counterparts and keep identical public
names; the two lineages evolve independently. The model is D6-invariant by
construction, so training uses no symmetry augmentation.

## Module table -- Python (`python/hexgnn/`)

| File | Role |
| --- | --- |
| `__init__.py` | Public surface re-exports (network, collate, config, constants, d6, features, losses). |
| `constants.py` | Node/edge-type ids, 32-wide D6-invariant node-feature slot layout (FEATURE_SCHEMA_VERSION 4, sparse rewrite), candidate-radius decisions, `feature_slots_after()` for zero-init resume. Mirrored by `rust/src/constants.rs`. |
| `d6.py` | 12-element hex D6 group (transform/inverse/compose), used by equivariance tests; the model is D6-invariant by construction, so no augmentation. |
| `features.py` | Python featurizer: `rust_bridge.graph_facts` dict -> `GraphTensors`. Parity twin of the Rust featurizer (must stay byte-identical). |
| `collate.py` | Packs variable-size `GraphTensors` into one disjoint-graph batch dict (the `HexgnnNetwork.forward` contract); deterministic. |
| `graph_build.py` | Glue: live engine state -> `graph_facts` -> featurize -> collate. Used by tests, `expand.py`, and `inference.evaluate_states`. |
| `architecture.py` | `HexgnnNetwork`: shared node-in projection, `RelationalMessagePassing` trunk, optional `SteerableTensorChannels`, `PMAValuePool` value readout, policy/opp heads; zero-init resume helpers; `precompute_side_rows` for torch.compile. |
| `config.py` | `parse_hexgnn_config` TOML boundary -> frozen dataclasses; rejects unknown keys. |
| `losses.py` | 65-bin value loss (reused from dense_cnn) + segmented per-graph softmax CE for the dynamic candidate policy + `hexgnn_loss` aggregator. |
| `trainer.py` | `HexgnnTrainer`: AdamW + AMP GradScaler + grad clip, warmup + resume-safe LR decay, transient-CUDA retry guard, `train_on_shards` via dense_cnn `compact_io`. |
| `expand.py` | Recompute-at-expand: compact .npz rows -> engine replay -> Rust graph + policy/opp/value targets, with BC-dataset pruning by out-of-candidate visit mass. |
| `checkpoints.py` | Checkpoint loader/saver for the plugin/config-CLI path (`{"model": "hexgnn", "model_state": ...}` + `.txt` pointer indirection). The RL driver uses its own format (see Checkpoint formats). |
| `plugin.py` | `HexgnnPlugin` for the `hexo_train` registry (build_model, component overrides, generate_selfplay, evaluate_epoch) -- the dormant config-CLI path. |
| `rust_bridge.py` | Thin boundary to `hexo_models._rust.hexgnn` (capabilities, candidate_ids, graph_facts, MCTS session). Readable error if the native module is absent. |
| `mcts.py` | `HexgnnMctsSession` Python wrapper (subtree reuse keyed by game, per-root PCR overrides) + byte-backed `CompactVisitPolicy`/`SearchResult` decode. |
| `inference.py` | `HexgnnInference` fp16 evaluator: pad-budget sorted chunking, pinned-memory H2D, int32 wire narrowing, nan/inf sanitization audit; `evaluate_featurized_batch` is the Rust MCTS callback. |
| `player.py` | `hexo_runner` player adapter (deterministic greedy eval, optional opening temperature); used by `evaluation.make_hexgnn_factory`. |
| `evaluation.py` | `run_head_to_head` (sequential; used by the RL driver and the `evaluate_epoch` SealBot hook) plus a batched variant, `HexgnnBatchedSearcher` / `run_head_to_head_parallel`, mirroring the hexgt API. |
| `selfplay.py` | Game-driven self-play: batched MCTS over active games, KataGo PCR full/fast split, temperature schedules, policy-surprise row duplication, soft-Z targets, sanitization-taint exclusion, `.hxr` records + dense_cnn-format compact shards. |

## Module table -- Rust (`rust/src/`)

The crate is not a standalone Cargo package (no Cargo.toml here). It is
`#[path]`-included by `packages/hexo_models/rust/src/lib.rs` and compiled into
the single native module as the submodule `hexo_models._rust.hexgnn`.

| File | Role |
| --- | --- |
| `lib.rs` | Crate root: registers candidates/features/mcts/threats/vcf pyfunctions + `capabilities()` metadata. |
| `constants.rs` | Rust mirror of the feature layout + search bounds (eval chunk, cache size, active-root limit). |
| `state.rs` | Clones live `hexo_engine.HexoState` via the versioned C-ABI capsule (`state_api`, version 2). |
| `candidates.rs` | Candidate set (active windows union n-radius minus dead cells) + bounded typed-graph builder, shared by sample-gen and live MCTS. |
| `features.rs` | Rayon-parallel featurizer + collator emitting zero-copy buffer-protocol batches; must stay byte-identical to `features.py`/`collate.py`. |
| `mcts.rs` | `HexgnnMctsSession` pyclass: batched PUCT search over many games, subtree promotion/reuse, per-root PCR overrides. |
| `mcts_tree.rs` | Model-agnostic PUCT tree mechanics (nucleus widening, forced playouts, Dirichlet noise) + TSS tactical-candidate injection. |
| `mcts_eval.rs` | Evaluator boundary: state hashing/transposition cache, in-Rust graph construction for leaves, calls the Python evaluator, parses the values/priors byte contract. |
| `threats.rs` | Phase-aware Connect6 threat / hitting-set analysis used by the tree injection and leaf-value override. A lineage-local fork; the dense_cnn/hexgt crates use the shared `crate::threats_shared` instead. |
| `vcf.rs` | Exploratory depth-bounded forcing-move (VCF) solver prototype, kept as a benchmark artifact; not wired into the live MCTS. |

## Packaging and build

hexgnn's Python is a plain setuptools package (`pyproject.toml`, package dir
`python/`). Its Rust lives in this package's `rust/src` but is compiled by the
`hexo_models` maturin build: `hexo_models/rust/src/lib.rs` `#[path]`-includes
`../../../hexgnn/rust/src/lib.rs`, and the `hexo_models` sdist bundles
`../hexgnn/rust`. Installing hexgnn therefore never builds native code; Rust
changes take effect after rebuilding `hexo_models`
(`scripts/_rebuild_hexo_models_hexgt.sh`, WSL venv). The package's integration
points outside its own directory are that `#[path]` include, the sdist glob in
`hexo_models/pyproject.toml`, and the `hexo_train.models` entry point declared
in this package's `pyproject.toml`.

## Checkpoint formats

Each entry path has its own checkpoint format:

- The RL driver (`scripts/_rl_train_hexgnn.py`) saves and resumes
  `{"model": <state_dict>, "arch": <meta>, "optimizer": ..., "train_state": ...}`.
  This is the format `hexo_frontend/debug_infer.py` loads as the graph (HEXGT)
  lineage for the dashboard debug screen.
- The plugin/config-CLI path (`checkpoints.py`) saves
  `{"model": "hexgnn", "model_state": ...}` behind a `.txt` pointer file.

## Connections to other packages

Imports OUT (what hexgnn depends on):

- `hexo_models._rust.hexgnn` -- via `rust_bridge.py`. The native code lives in
  this package's `rust/` tree but is compiled into the `hexo_models` wheel
  (see Packaging and build).
- `hexo_models.dense_cnn` -- `selfplay.py` writes and `trainer.py`/`expand.py`
  read dense_cnn's compact .npz shard format (`compact_io.write_compact_shard`
  / `read_compact_shard`); sample finalization reuses `dense_cnn.samples`
  (`Model1SampleData`, `finalize_game_samples`, `sample_from_state`) and
  `dense_cnn.replay.materialize_policy_surprise_rows`.
- `hexo_engine` -- `expand.py` replays placement histories; selfplay/evaluation
  drive live states; Rust `state.rs`/`candidates.rs` consume `HexoState` and
  the `WindowStore` directly.
- `hexo_runner` -- `player.py` implements the runner player lifecycle;
  `evaluation.py` uses `hexo_runner.adapters.sealbot.SealBotPlayer` and
  `hexo_runner.modes.match.run_match`; `selfplay.py` writes `.hxr` records via
  `hexo_runner.records.HexoRecordFile`.
- `hexo_train` -- `plugin.py` implements the plugin contract
  (`hexo_train.components.ComponentOverrides`) and is registered under the
  `hexo_train.models` entry-point group as `hexgnn` (this package's
  pyproject.toml).

Imports IN / consumers:

- `packages/hexo_models/rust/src/lib.rs` `#[path]`-includes
  `../../../hexgnn/rust/src/lib.rs` (so every hexo_models native build
  compiles this crate), and the hexo_models sdist bundles `../hexgnn/rust`.
- `hexo_frontend/web.py` special-cases hexgnn/hexgt self-play diagnostics
  fields for the dashboard run-history views; `debug_infer.py` loads the
  driver-format checkpoints as the graph (HEXGT) lineage.
- `_dashboard_bridge_hexgnn.py` (repo root) mirrors `runs/hexgnn_rl_main1`
  outputs into the :8080 dashboard layout.

Protocols:

- Evaluator byte protocol: Rust `mcts_eval.rs` calls
  `HexgnnInference.evaluate_featurized_batch` with a zero-copy
  buffer-protocol collated batch and expects back `{values_bytes,
  priors_bytes}` float32 in Rust's packed candidate order.
- Engine state intake: `hexo_engine._rust.state_api_capsule()` C-ABI capsule,
  STATE_API_VERSION 2; version mismatch fails loudly at use time.

## Entry points / how it gets exercised

| Entry | What it does |
| --- | --- |
| `scripts/_rl_train_hexgnn.py` | Primary RL driver (BC-seeded selfplay -> train -> eval loop); imports the package directly with a PYTHONPATH self-bootstrap, bypassing the plugin. |
| `scripts/_pretrain_hexgnn.py` | Behavior-cloning pretrain producing the seed checkpoint the RL driver consumes. |
| `scripts/_rl_launch_hexgnn.sh` / `scripts/_rl_supervise_hexgnn.sh` | WSL setsid launch + crash-restart supervisor (also starts the dashboard bridge). |
| `_dashboard_bridge_hexgnn.py` | Read-only dashboard mirror loop for `runs/hexgnn_rl_main1`. |
| `configs/hexgnn_model.toml` + the `hexo_train.models` entry point | Dormant config-CLI path: `python -m hexo_train.cli.train_model configs/hexgnn_model.toml`. |
| `tests/test_hexgnn_*.py` (9 files) | model, losses, d6, selfplay, compile, steerable, value_readout, eval_identity, featurizer_parity. Run in the WSL venv, where the native module is built. |
