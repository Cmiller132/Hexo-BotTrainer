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
its Rust crate is still compiled into every `hexo_models` native build.

## Module table — Python (`python/hexgnn/`)

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
| `checkpoints.py` | Plugin-path checkpoint loader/saver (`{"model": "hexgnn", "model_state": ...}` + `.txt` pointer indirection). NOTE: the actual RL driver uses a different format (see gotchas). |
| `plugin.py` | `HexgnnPlugin` for the `hexo_train` registry (build_model, component overrides, generate_selfplay, evaluate_epoch) — the dormant config-CLI path. |
| `rust_bridge.py` | Thin boundary to `hexo_models._rust.hexgnn` (capabilities, candidate_ids, graph_facts, MCTS session). Readable error if the native module is absent. |
| `mcts.py` | `HexgnnMctsSession` Python wrapper (subtree reuse keyed by game, per-root PCR overrides) + byte-backed `CompactVisitPolicy`/`SearchResult` decode. |
| `inference.py` | `HexgnnInference` fp16 evaluator: pad-budget sorted chunking, pinned-memory H2D, int32 wire narrowing, nan/inf sanitization audit; `evaluate_featurized_batch` is the Rust MCTS callback. |
| `player.py` | `hexo_runner` player adapter (deterministic greedy eval, optional opening temperature); used by `evaluation.make_hexgnn_factory`. |
| `evaluation.py` | `run_head_to_head` (sequential; used by the RL driver and `evaluate_epoch` SealBot hook) plus `HexgnnBatchedSearcher`/`run_head_to_head_parallel` (no live callers found). |
| `selfplay.py` | Game-driven self-play: batched MCTS over active games, KataGo PCR full/fast split, temperature schedules, policy-surprise row duplication, soft-Z targets, sanitization-taint exclusion, `.hxr` records + dense_cnn-format compact shards. |

## Module table — Rust (`rust/src/`)

The crate is NOT standalone (no Cargo.toml here). It is `#[path]`-included by
`packages/hexo_models/rust/src/lib.rs` and compiled into the single native
module as the submodule `hexo_models._rust.hexgnn`.

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
| `threats.rs` | Phase-aware Connect6 threat / hitting-set analysis used by the tree injection and leaf-value override. This is a fork, not the shared `crate::threats_shared` used by dense_cnn/hexgt. |
| `vcf.rs` | Exploratory depth-bounded forcing-move (VCF) solver prototype; self-documented as a benchmark artifact, NOT wired into live MCTS. |

## Connections to other packages

Imports OUT (what hexgnn depends on):

- `hexo_models._rust.hexgnn` — via `rust_bridge.py`. The native code lives in
  this package's `rust/` tree but is compiled into the `hexo_models` wheel;
  installing hexgnn never rebuilds the native module.
- `hexo_models.dense_cnn` — `selfplay.py` writes and `trainer.py`/`expand.py`
  read dense_cnn's compact .npz shard format (`compact_io.write_compact_shard`
  / `read_compact_shard`); sample finalization reuses `dense_cnn.samples`
  (`Model1SampleData`, `finalize_game_samples`, `sample_from_state`) and
  `dense_cnn.replay.materialize_policy_surprise_rows`.
- `hexo_engine` — `expand.py` replays placement histories; selfplay/evaluation
  drive live states; Rust `state.rs`/`candidates.rs` consume `HexoState` and
  the `WindowStore` directly.
- `hexo_runner` — `player.py` implements the runner player lifecycle;
  `evaluation.py` uses `hexo_runner.adapters.sealbot.SealBotPlayer` and
  `hexo_runner.modes.match.run_match`; `selfplay.py` writes `.hxr` records via
  `hexo_runner.records.HexoRecordFile`.
- `hexo_train` — `plugin.py` implements the plugin contract
  (`hexo_train.components.ComponentOverrides`) and is registered under the
  `hexo_train.models` entry-point group as `hexgnn` (pyproject.toml).

Imports IN / consumers:

- `packages/hexo_models/rust/src/lib.rs` `#[path]`-includes
  `../../../hexgnn/rust/src/lib.rs` (so every hexo_models native build
  compiles this crate), and the hexo_models sdist bundles `../hexgnn/rust`.
- `hexo_frontend/web.py` special-cases hexgnn/hexgt self-play diagnostics
  fields for the dashboard run-history views; `debug_infer.py` handles the
  driver's `{'model': state_dict, 'arch': meta}` checkpoint format as the
  graph (HEXGT) lineage.
- `_dashboard_bridge_hexgnn.py` (repo root) mirrors `runs/hexgnn_rl_main1`
  outputs into the :8080 dashboard layout.

Protocols:

- Evaluator byte protocol: Rust `mcts_eval.rs` calls
  `HexgnnInference.evaluate_featurized_batch` with a zero-copy
  buffer-protocol collated batch and expects back `{values_bytes,
  priors_bytes}` float32 in Rust's packed candidate order. (Some docstrings
  still name a nonexistent `evaluate_graph_facts` — see gotchas.)
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
| `tests/test_hexgnn_*.py` (9 files) | model, losses, d6, selfplay, compile, steerable, value_readout, eval_identity, featurizer_parity. Authoritative only in the WSL venv. |

## Gotchas

- **Stale packaging story.** `pyproject.toml` claims hexgnn "carries NO Rust of
  its own" and "reuses `hexo_models._rust.hexgt`". Both claims are outdated:
  the package has its own `rust/` crate, compiled as `hexo_models._rust.hexgnn`.
  The same stale claim appears in `configs/hexgnn_model.toml` and the
  `_rl_train_hexgnn.py` docstring.
- **Rust still builds even though the package is parked.** Retiring hexgnn
  means removing the `#[path]` include in `hexo_models/rust/src/lib.rs`, the
  sdist glob in `hexo_models/pyproject.toml`, and the entry point — not just
  the package directory.
- **Checkpoint-format split-brain.** The plugin path (`checkpoints.py`) saves
  `{'model': 'hexgnn', 'model_state': ...}`; the actual RL driver saves
  `{'model': state_dict, 'arch': meta}`. The frontend lineage sniffer only
  recognizes the driver format as the graph lineage; a plugin-saved hexgnn
  checkpoint would be misclassified as DENSE_RESTNET.
- **Dangling docstrings.** `mcts.py` and `rust/src/mcts_eval.rs` reference
  `HexgnnInference.evaluate_graph_facts` (does not exist; the real callback is
  `evaluate_featurized_batch`); `rust/src/features.rs` cites a guard test
  `tests/test_hexgnn_feature_buffer.py` that does not exist (the actual parity
  test is `tests/test_hexgnn_featurizer_parity.py`); `rust_bridge.py` says the
  graph builders "are added as they land" though they already exist.
- **Stale architecture rationale.** `architecture.py`'s docstring justifies
  `value_head_use_side` via the `EDGE_TYPE_CONTEXT` hub, but `constants.py`
  marks that edge type RETIRED after the sparse rewrite; the SIDE node is now
  edge-isolated.
- **Copy divergence with hexgt.** evaluation/inference/mcts/player and the
  whole Rust tree are copied-verbatim twins of `hexo_models/hexgt` with
  identical public names. Fixes to one lineage do not propagate; if hexgnn is
  ever revived, diff-audit against hexgt first.
- **Known-uncalled surface** (kept as API mirrors of hexgt):
  `evaluation.run_head_to_head_parallel` / `HexgnnBatchedSearcher`,
  `rust_bridge.capabilities()` / `candidate_ids()`, the Python-facing
  `hexgnn_threat_analysis` pyfunction, and `rust/src/vcf.rs`
  (`hexgnn_vcf_solve`).
- **Soft validation bound.** `constants.py` `MAX_CANDIDATE_RADIUS = 4` is
  documented as soft/unused; only Rust enforces `candidate_radius >= 1`, and
  `config.py` performs no range checks on it.
- `_dashboard_bridge_hexgnn.py` comments still call `runs/hexgnn_rl_main1`
  "the ACTIVE run" — it is not.
