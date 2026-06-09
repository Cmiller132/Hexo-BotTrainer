# hexgnn — transformer-free hexgt lineage

`hexgnn` is a **stripped-down copy of hexgt** (Model 2/3): it keeps hexgt's
relational-GNN message-passing trunk and the **full TSS / self-play / replay /
eval stack**, but **removes the context transformer and the short-term-value
(STV) lookahead heads**. The heads are **policy, value, and opponent-policy only**.

It is **purely additive** and does not disturb the live `hexgt_rl_main3` run:

- It carries **no Rust of its own**. It reuses the already-built native
  accelerator inside `hexo_models` — `hexo_models._rust.hexgt` — **read-only**
  (the candidate/window/graph builder, the Rust featurizer, the batched PUCT MCTS
  with nucleus widening, and the WindowStore threat index). Because the feature /
  candidate / graph layout is byte-identical, **featurizer parity carries over**
  (gated by `tests/test_hexgnn_featurizer_parity.py`) and all the **TSS coupling**
  (tactical injection at expansion, phase-aware hitting-set leaf overrides,
  tactical move-selection guard) is inherited unchanged.
- The Python halves of the trunk, MCTS boundary, self-play, trainer, replay,
  inference, eval, and checkpoints are copied from hexgt and stripped of the
  transformer + STV plumbing.

## What changed vs hexgt

| | hexgt model-3 | hexgnn |
|---|---|---|
| GNN trunk (relational msg passing) | 4 layers | 4 layers (same) |
| context transformer | 3 layers | **removed** |
| heads | policy, value, opp, STV[4,12,24] | **policy, value, opp** |
| value readout | `[SIDE \| PMA_2]` | `[SIDE \| PMA_2]` (kept — see below) |
| params | 2,584,774 | **931,459** (−64%) |

### Value-readout decision — kept the PMA pool
Without the transformer there is no attention mixing into the SIDE hub, so the
value readout **must pool globally over the post-GNN node embeddings**. We kept
the Set-Transformer **PMA** pool (k=2 seeds, varlen scatter-softmax over all
nodes) rather than a fixed mean/max pool: it *is* that global pool, it is already
proven / D6-invariant / `torch.compile`-clean in this codebase, and it is a strict
generalization of mean+max at the cost of one small module. The SIDE hub is still
concatenated (it carries GNN-propagated global state via the context-hub edges).
See the docstring in `python/hexgnn/architecture.py`.

## Layout
`python/hexgnn/` mirrors `hexo_models/hexgt/python/hexo_models/hexgt/` with the
transformer/STV removed. `rust_bridge.py` points at the shared native
`hexo_models._rust.hexgt`.

## Use (CPU dev or GPU run)

The package is importable via PYTHONPATH (no install needed), or `pip install -e
packages/hexgnn` to register the `hexo_train.models` entry point for the
config-driven CLI.

```
# 1) pretrain (behavioral clone) from an existing buffer (READ-ONLY):
python scripts/_pretrain_hexgnn.py --buffer runs/hexgt_rl_main3/selfplay \
    --out runs/hexgnn_rl/pretrain/hexgnn_pretrain.pt --device cuda

# 2) launch the RL run (detached supervisor):
bash scripts/_rl_launch_hexgnn.sh
# or run the driver directly:
python scripts/_rl_train_hexgnn.py --bc-seed runs/hexgnn_rl/pretrain/hexgnn_pretrain.pt \
    --out-dir runs/hexgnn_rl --epochs 60 --sealbot
```

Tests: `tests/test_hexgnn_*.py` (CPU-only; forward shapes, D6 equivariance,
featurizer parity, value readout, losses, torch.compile, end-to-end mini
self-play).
