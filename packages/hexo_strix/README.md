# hexo_strix

A dependency-light **port of the [SootyOwl/hexo-strix](https://github.com/SootyOwl/hexo-strix) GNN bot** so one of its checkpoints can be run as an **evaluation opponent** on this repo's game engine + runner.

hexo-strix plays the same game as this repo (infinite hex tic-tac-toe, win-length 6, two placements per turn) but with a completely different net: a **graph neural network** (`HeXONet` — a GINE trunk over an "axis" graph, JumpingKnowledge-cat, per-node policy + stone-pooled value) driven by Gumbel MCTS. This package reproduces the network and its board→graph featurizer in **pure PyTorch / Python** — no `torch_geometric` (unavailable on this machine's Python 3.14 / torch 2.10) and no hexo-strix Rust extension.

## What it does

- **`model.py`** — `HeXONet` reimplemented so parameter names match the upstream state dict exactly; loads `checkpoint_*.pt` with `strict=True`. Custom `GINEConvLite` reproduces `torch_geometric.nn.GINEConv` semantics.
- **`graph.py`** — verbatim port of the Rust "axis" graph builder (`axis_graph.rs`), threat features (`threat.rs`), legal-move enumeration and hex distance. Engine-agnostic (players are `+1`/`-1` ints) so it is unit-testable against hexo-strix's own Rust test vectors.
- **`loader.py`** — `load_strix_checkpoint(path)` → eval-ready model + embedded config.
- **`player.py`** — `StrixPlayer`, a `hexo_runner` `RunnerPlayer`. One placement per `decide()` (the engine turn is autoregressive), **raw-policy greedy** move selection (argmax over the net's legal-node logits — deterministic, no MCTS), so it is a fixed-strength anchor.

## Two eval opponents

### 1. `StrixMctsPlayer` — the faithful eval (recommended)

Runs hexo-strix's **actual Gumbel-AlphaZero MCTS**: the compiled Rust `hexo_rs`
extension (`gumbel_mcts_with_diagnostics`), built from the upstream repo, driving
the network through a Python callback — then plays `legal_moves[argmax(improved_policy)]`,
exactly as `hexo_a0.evaluate.play_eval_game`. This *is* their eval algorithm.

The network callback uses this repo's ported model + graph builder, which was
verified **numerically identical** to hexo-strix's own torch_geometric model +
Rust graph builder (max |Δlogit| = 0, max |Δvalue| ≈ 6e-8). So the search runs
their exact algorithm over identical priors/values — with only the `hexo_rs`
wheel as an extra runtime dep (no torch_geometric, no hexo_a0). See
[vendor/BUILD_hexo_rs.md](vendor/BUILD_hexo_rs.md) to install/rebuild it.

Eval search defaults match hexo-strix's `EvalConfig` (eval games):
**256 sims, m_actions=16, c_visit=50, c_scale=1.0**. Gumbel root noise is disabled
by default for reproducible paired eval (the paper's deterministic eval mode);
pass `disable_gumbel_noise=False` (+ a seed) to mirror their stochastic default.

```python
from hexo_strix import make_strix_mcts_factory
factory = make_strix_mcts_factory("checkpoint_00237000.pt", device="cpu", sims=256)
# factory(seed) -> StrixMctsPlayer, a RunnerPlayer for run_head_to_head
```

### 2. `StrixPlayer` — raw-policy greedy (fast fallback)

Move = `argmax(policy_logits)` (hexo-strix's `mcts_sims=0` path). Deterministic,
needs no `hexo_rs`, but weaker than the full search. Use for quick checks.

```python
from hexo_strix import make_strix_factory
factory = make_strix_factory("checkpoint_00237000.pt", device="cpu")
```

## Fidelity notes (both paths)

- **Candidate cells use hexo-strix's `placement_radius` (6, from the checkpoint's `game_config`)**, a subset of this engine's radius-8 legal set — so every move is legal here (validated with `engine.is_legal_action`), and the graph the net sees matches its training distribution. The opening move is played at the origin `(0,0)`.
- Player mapping: this engine's `player0` (first mover, opens at origin) ↔ hexo-strix `P1`.
- A HeXO turn is two placements; the tree searches one placement per node and eval runs a separate search per placement — matching this repo's one-placement-per-`decide` runner, so no buffering is needed.

Head-to-head / smoke driver: [`scripts/_strix_h2h.py`](../../scripts/_strix_h2h.py)

```bash
# real Gumbel MCTS (default), 256 sims:
python scripts/_strix_h2h.py --ckpt ~/Downloads/checkpoint_00237000.pt --vs random --search mcts --sims 256 --games 10
python scripts/_strix_h2h.py --vs sealbot --search mcts   # needs a SealBot checkout (SEALBOT_PATH)
python scripts/_strix_h2h.py --vs random --search policy   # raw-policy fallback (no hexo_rs)
```

To fold into the generic eval routine, pass `make_strix_factory(ckpt)` as an
opponent to `run_head_to_head` from any model package's `evaluation` module
(e.g. `hexo_models.hexgt.evaluation`), exactly like `scripts/_head_to_head.py`.

> Note: this bot cannot enter `hexfield.multistage_eval` / `eval_arena` — that
> arena is hard-wired to hexfield checkpoints + the Rust MCTS evaluator
> protocol. The generic `run_head_to_head` path is the integration surface.

## Verification

`tests/test_hexo_strix_port.py`:
- threat features match hexo-strix's Rust unit-test vectors exactly;
- the checkpoint loads with `strict=True` (architecture is exact);
- given 5-in-a-row with open ends, the bot plays a winning completion.

Confirmed end-to-end through this repo's runner: beats a random baseline 6-0 and
plays deterministic, legal, sensible games.

## Environment caveat

This repo's `hexo_runner` editable install is currently broken on the Windows
Python 3.14 interpreter (it points at the sibling `E:\Hexo-BotTrainer` checkout).
`scripts/_strix_h2h.py` prepends the local `packages/hexo_runner/python` to
`sys.path` to work around it; `hexo_engine`, `hexo_utils`, and `hexo_models`
keep their installed compiled builds. Running `pip install -e packages/hexo_runner`
would repair the install globally (repointing it at this worktree).
