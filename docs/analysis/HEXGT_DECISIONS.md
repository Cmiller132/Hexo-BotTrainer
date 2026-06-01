# HEXGT (Model 2) — Implementation Decisions Log

Running log of non-trivial design decisions made while building `hexgt` (the
dynamic GNN + transformer Model 2) per `HEXFORMER_REWRITE_PLAN.md`. Each entry
records the decision, the plan reference, and the rationale. The plan's §11
"Open questions" and §12 readiness-gaps (A–J) are resolved here as implemented.

---

## Environment / safety (build isolation)

**Live-run isolation.** The dense_cnn 96×8 run is live on the GPU, editable-
installed (`hexo-models`) against **this same worktree** (`hexo_models.pth` →
`/mnt/e/Hexo-BotTrainer/packages/hexo_models/python`), with a supervisor
(`scripts/supervise_target_96x8_wsl.sh`) that can relaunch it and a dashboard
(`hexo_frontend.web`, port 8080). The only shared mutable artifact that could
break the live run is the in-place native module
`packages/hexo_models/python/hexo_models/_rust.cpython-312-*.so`.

- **Edits + commits** happen on the main worktree `/mnt/e/Hexo-BotTrainer`
  (branch `bench/inference-backends-wsl`). Safe for the live run: dense_cnn never
  imports `hexformer_ar` or `hexgt`; the running process holds its modules in
  memory; `hexo_models/__init__.py` guards sibling roots with `.is_dir()`.
- **Native builds + test runs** happen ONLY in an isolated worktree
  `/mnt/e/Hexo-BotTrainer-hexgtbuild` with its own venv `hexgt-build`, so the
  main tree's `_rust.so` is never rebuilt/overwritten while the live run depends
  on it. We never `maturin develop`/`pip install` into the live venv or the main
  worktree.
- Git commands run through Windows git (Git Bash) per the CRLF instruction.

---

## Phase 0 — delete + scaffold

**Package name (open-q #1): `hexgt`** (Hex Graph-Transformer). Accepted from the
plan working name.

**Deletion scope (§1).** Remove `packages/hexo_models/hexformer_ar/` (Py+Rust),
its `pyproject.toml` entry point + maturin include + pyright extraPath, the
`#[path]` include + `sys.modules` registration in
`packages/hexo_models/rust/src/lib.rs`, the sibling-root entries in
`packages/hexo_models/python/hexo_models/__init__.py`, `configs/hexformer_ar.toml`,
and `tests/test_hexformer_ar_*.py`. The `namespace="hexformer_ar"` string in
`tests/test_hexo_utils_sample_store.py` is a generic label for the hexo_utils
sample store (not an import) — left as-is to minimize churn (revisit if noisy).

**Scaffold mirrors dense_cnn's module set.** `hexgt/python/hexo_models/hexgt/`
with `__init__.py`, `constants.py`, `config.py`, stub `architecture.py`,
`losses.py` (reused 65-bin binning math), `checkpoints.py` (dense_cnn-pattern,
rejects `sample_buffer`), `plugin.py` (ModelPlugin), registered via the
`hexo_train.models` entry point as `hexgt`. Rust crate dir
`hexgt/rust/` added in Phase 1 (no Rust needed for the Phase-0 gate).

### Packed-graph batch contract (gap I) — provisional, locked enough for Phase 0/3

A batch packs `G` position-graphs into one disjoint graph (PyG-style). Tensors:

| key | shape | dtype | meaning |
|---|---|---|---|
| `node_type` | (Ntot,) | int64 | node type id (SIDE=0, STONE=1, CANDIDATE=2, WINDOW=3) |
| `node_feat` | (Ntot, F) | float32 | unified per-node feature vector (F = `NODE_FEATURE_DIM`) |
| `node_graph` | (Ntot,) | int64 | graph id per node (segment/block-diagonal) |
| `edge_index` | (2, Etot) | int64 | (src, dst) in packed node indexing |
| `edge_type` | (Etot,) | int64 | edge type id (see constants) |
| `candidate_index` | (Ctot,) | int64 | node indices that are candidates, in CSR/legal order |
| `candidate_graph` | (Ctot,) | int64 | graph id per candidate (per-graph policy softmax) |
| `num_graphs` | scalar | int | G |

Forward output:
- `policy`: (Ctot,) one logit per candidate (dynamic). Per-graph softmax in loss
  via `candidate_graph` segments → emitted as `priors_bytes` in legal order.
- `value`: (G, 65) 65-bin distributional (reuses dense_cnn binning).
- `opp_policy`: (Ctot,) per-candidate aux logits.
- `stvalue_<h>`: (G, 65) optional.

This maps directly onto the Rust per-move CSR priors contract (`legal_row_offsets`
order == `candidate_index` order within a graph).

### Node feature layout (gap A) — provisional, finalized in Phase 4 expand

`NODE_FEATURE_DIM = 32`, unified vector with type-routed slots (see
`constants.py` for named offsets). Provisional; the exact encoding is locked when
`expand_row_to_graph` (Phase 4) and the GNN input projections (Phase 3) are
written. The contract only requires builder and model to agree on `F`.

### Edge types (gap I/§6.3) — bounded, window-hub routed (NO same-axis cliques)

`ADJACENCY=0` (node↔node hex-dist 1, ≤6/node), `STONE_WINDOW=1` (window↔its
one-color stones, ≤6/window), `CANDIDATE_WINDOW=2` (window↔its empty cells,
≤6/window), `RECENCY=3` (stone↔prev/next, chain), `CONTEXT=4` (side/goal↔all,
1 hub). Stored symmetric (both directions emitted with the same type) for the
MVP. Total edges linear in (#nodes + #windows) — the §4.6 no-explosion gate.

### Resolved gaps from the user

- **(B) two-stone turns → EVERY STONE PLACEMENT is a separate move**, per-
  placement actions, sequential, exactly like dense_cnn. The policy/MCTS action
  is a single stone; `opp_policy` = the next placement decision (which may be the
  same player's second stone or the opponent's, per engine turn phase).
- **(C) value / short-term-value / opp_policy targets** mirror dense_cnn's
  `finalize_game_samples`, adapted to per-placement turns + candidate rep
  (detailed in Phase 4).
- **(A) node features, (D) D6 edge-relabel, (H) GNN/transformer hypers ~2.1M,
  (I) collation** — designed here / in later phase entries.
