# hexo_models/hexgt -- "Model 2/3" GNN + transformer lineage

Dynamic typed-graph neural network with a context-transformer stage and a
Rust batched PUCT MCTS, for the Hexo game (Connect6-style, unbounded hex
grid). Owns its full vertical slice: Rust candidate/graph builder +
featurizer + search, and the Python config/trainer/self-play/eval/checkpoint
stack, plugged into the generic `hexo_train` pipeline as the `hexgt` plugin.

## Status

Legacy as a training lineage: its RL run, `hexgt_rl_main3`, halted
2026-06-05, and the active training lineage is `packages/dense_cnn_restnet`.
The code remains live infrastructure:

| Consumer | Use |
|---|---|
| `packages/hexgnn` | A stripped-down fork of this package; mirrors its Python modules and compiles into the same native module. |
| `packages/hexo_frontend/python/hexo_frontend/debug_infer.py` | Dashboard debug worker and Match Arena load hexgt checkpoints through this package. |
| `packages/dense_cnn_restnet` | Adopted this package's PCR (Playout Cap Randomization) design -- design lineage, no import. |
| `tests/test_hexgt_*.py` (~30 files) | Contract gates: Rust/Python featurizer byte parity, D6 equivariance, MCTS/PCR/TSS regressions. Run in the WSL venv (the native module is Linux-only). |
| `hexo_models` native build | The Rust crate here is `#[path]`-included into the single `hexo_models` cdylib; every rebuild compiles it. |

Runs that used this lineage: the model2/model3 configs
(`configs/hexgt_model2.toml`, `configs/hexgt_model3.toml`) and the RL run
`hexgt_rl_main3`, driven by `scripts/_rl_train.py` (BC-seeded
selfplay -> train -> eval loop) under `scripts/_rl_supervise.sh` /
`scripts/_rl_launch_main3.sh`, with BC seeding via `scripts/_bc_train.py` /
`scripts/_pretrain_model3.py`.

## Architecture (python/hexo_models/hexgt/architecture.py)

`HexgtNetwork` -- a GNN/transformer hybrid over a *dynamic* per-position
graph (candidate cells, stones, window-hub nodes, one SIDE hub), packed as
one disjoint batch by `collate.py`. All node/edge features are D6-invariant
(`features.py`) and all ops permutation-equivariant, so the model is
D6-invariant **by construction** -- no symmetry augmentation is used.

Pipeline (defaults: `token_dim=168`, 3 GNN layers, 3 transformer layers,
4 attention heads, `ffn_dim=336`):

1. `node_in`: shared 2-layer MLP projecting 32-slot node features.
2. `RelationalMessagePassing` x N: typed message passing -- per-edge-type
   weight tensor applied via einsum, mean aggregation, residual + LayerNorm.
   Line/co-linearity relations are pre-routed through window-hub nodes by the
   Rust graph builder (no same-axis cliques).
3. `GraphTransformerLayer` x N: per-graph context self-attention over
   {side, stone, window} tokens + candidate -> context cross-attention,
   batched across graphs via a precomputed padded `_AttentionLayout`
   (`precompute_attention_layout` exists for the torch.compile path).
4. Heads:

| Head | Shape | Notes |
|---|---|---|
| `policy` | (Ctot,) | One logit per candidate node; variable length per graph. |
| `value` | (G, 65) | 65-bin value distribution. Readout = `[SIDE \| PMA_k]`: SIDE hub embedding + a Set-Transformer PMA pool (k learned seed queries, default k=2) over all post-transformer nodes. |
| `opp_policy` | (Ctot,) | Aux: opponent's reply policy per candidate. |
| `stvalue_<h>` | (G, 65) | Short-term-value heads per horizon; consume the same `[SIDE \| PMA_k]` readout. |

The module also carries warm-start grafts (`zero_init_expanded_feature_columns`,
`expand_value_readout_columns`, `expand_stv_readout_columns`) used by
`scripts/_rl_train.py` and the frontend to load checkpoints across feature
schema / readout-shape changes. Losses live in `losses.py` (65-bin value CE
reused from dense_cnn, segmented per-graph softmax CE for the dynamic policy).

## Rust MCTS (rust/src/)

The crate is NOT standalone: `packages/hexo_models/rust/src/lib.rs`
`#[path]`-includes `hexgt/rust/src/lib.rs`, exposing it to Python as
`hexo_models._rust.hexgt`. Rebuild with
`scripts/_rebuild_hexo_models_hexgt.sh` (maturin, WSL hexgt-build venv);
Rust changes take effect on the next rebuild.

| File | Role |
|---|---|
| `mcts.rs` | `HexgtMctsSession` PyO3 class: batched PUCT over many concurrent games, per-game-key promoted-subtree reuse across turns, shared transposition cache, virtual-loss batched leaf selection, root Dirichlet noise / policy temperature / forced playouts, per-root override knobs, TSS verdict short-circuit. |
| `mcts_tree.rs` | Model-agnostic PUCT tree: lazy edge materialization with nucleus (top-p) widening, FPU, virtual loss, tactical edge injection (forced visits for threat cells), subtree promotion. |
| `mcts_eval.rs` | Evaluator boundary: hash/dedupe leaf states, Rust-side featurize (no Python re-clone), chunked callback into the Python evaluator, FIFO eval cache. |
| `candidates.rs` | THE shared candidate set (active windows union n-radius minus dead cells, default n=3) + typed-graph builder used by both sample-gen and live search. |
| `features.rs` | Rayon-parallel featurizer/collator emitting a zero-copy buffer-protocol batch; byte-identical to `features.py`/`collate.py` by contract (parity-tested). |
| `threats.rs` | Shim re-exporting `crate::threats_shared` (single TSS threat/win-now/forced-loss definition shared with dense_cnn) + a diagnostic pyfunction. |
| `vcf.rs` | Depth-bounded forcing-move solver exposed as a benchmark hook; not wired into live search. |
| `state.rs` | Clones live `hexo_engine.HexoState` objects via the `hexo_engine._rust` PyCapsule state API (version 2). |

Evaluator protocol: Rust calls
`HexgtInference.evaluate_featurized_batch(batch)` (a collated-graph dict of
buffer-protocol arrays) and parses back `{values_bytes, priors_bytes}` --
float32 bytes, priors in the packed candidate CSR order. `inference.py`
implements this with FP16, VRAM-budgeted sorted chunking, and sanitization
of non-finite logits.

## Python session API (python/hexo_models/hexgt/mcts.py)

```python
from hexo_models.hexgt.mcts import new_mcts_session
from hexo_models.hexgt.inference import HexgtInference

session = new_mcts_session(max_states=1_048_576, n=3)  # n = candidate radius
results = session.run(
    game_keys, root_states, inference,   # one live engine state per game key
    visits=600, c_puct=1.5, temperature=1.0, seed=...,
    # root knobs: root_dirichlet_total_alpha/_noise_fraction,
    # root_policy_temperature, fpu_reduction, virtual_loss,
    # widening_policy_mass/_max_children/_min_children, forced_playout_k,
    # move_temperatures, active_root_limit, virtual_batch_size,
    # per_root_visits / per_root_forced_playout_k / per_root_noise,
)
```

- `game_keys` identify independent games; Rust promotes the selected child
  after each search and reuses the subtree under that key on the next turn
  (`discard(key)` / `clear()` to drop trees).
- Each `run()` returns one `SearchResult` per root: `action_id`,
  byte-backed lazily-decoded `visit_policy` / `root_prior_policy`
  (`CompactVisitPolicy`), `root_value`, `visits`, `diagnostics`.
- The `per_root_*` overrides allow heterogeneous visit caps / noise within
  one batched call (so PCR full+fast can share a forward stream); production
  selfplay issues two separate `run()` calls for the full and fast subsets.

## Self-play and PCR (python/hexo_models/hexgt/selfplay.py)

`run_selfplay_games(...)` drives many concurrent games through ONE MCTS
session, batching all due decisions per round. Key mechanics:

- **PCR (KataGo Playout Cap Randomization, Wu 2020).** Per move, a
  deterministic coin `_pcr_is_full(base_seed, epoch, game_key, move_index,
  full_proportion)` -- a splitmix64 hash, reproducible and decorrelated
  across games/moves/epochs -- chooses:
  - FULL search: `search_visits` cap, Dirichlet noise + forced playouts +
    temperature schedule, **recorded** as a training row;
  - FAST search: `pcr_fast_visits` cap, no noise, played greedily, **not**
    a policy/value target (fast rows are kept through finalization for the
    value/STV chain, then dropped; `mask_opp_from_fast` masks opp-policy
    targets that point at fast moves).
- **Policy-surprise weighting**: rows are duplicated proportionally to
  KL(visits || prior) via
  `hexo_models.dense_cnn.replay.materialize_policy_surprise_rows`.
- **Soft-Z value targets** (`samples.soft_z_lambda`) through
  `hexo_models.dense_cnn.samples.finalize_game_samples`.
- **Sanitization exclusion**: if the evaluator sanitized any non-finite
  logit in a batched round, every position decided that round is excluded
  from training data.
- Outputs: `.hxr` game records (`hexo_runner.records.HexoRecordFile`) +
  dense_cnn-format compact `.npz` shards
  (`hexo_models.dense_cnn.compact_io`), and a rich `SelfPlayResult`
  diagnostics dataclass (entropy, PCR counters, surprise stats).

## Connections

| Boundary | Contract |
|---|---|
| `hexo_train` | Entry point `hexgt = hexo_models.hexgt.plugin:get_plugin` (`packages/hexo_models/pyproject.toml`); `plugin.py` wires build_model / trainer / checkpoints / selfplay / eval. The `hexgt_rl_main3` run drove training through `scripts/_rl_train.py` directly rather than the plugin loop. |
| `hexo_models.dense_cnn` | Reuses `compact_io` (shard format), `samples` (compact rows + finalization), `replay.materialize_policy_surprise_rows`. |
| `hexo_engine` | Live states via the Python API; Rust side clones states through the v2 PyCapsule state API and reads the engine's incremental `WindowStore` for candidates/threats. |
| `hexo_runner` | `player.py` implements the runner player protocol (greedy eval play); `evaluation.py` runs SealBot gating via `run_match` + `SealBotPlayer`; selfplay writes `.hxr` via `hexo_runner.records`. |
| `hexo_utils` | `hash_state` (Rust) keys the evaluator/transposition caches. |
| Checkpoints | `.pt` payload `{model: "hexo_models.hexgt", model_state, optimizer_state, train_state, epoch, metadata}` with optional `.txt` pointer indirection (`checkpoints.py`); read by `scripts/_rl_train.py` and the frontend. |

Cross-language contract: `constants.py` + `features.py` and
`rust/src/constants.rs` + `features.rs` define the same featurization and are
held byte-identical by the parity tests.
