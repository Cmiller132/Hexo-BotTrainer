"""TSS + soft-value end-to-end smoke (Validation phase). Runs the FULL composed
pipeline on the free GPU: self-play (search exercises Phase-4 injection + Phase-5
override) -> soft-Z finalize (Phase 1) -> one training step (Phase-2 global-pooled
value head + Phase-3 v3 features). Self-contained: builds a fresh small model, does
NOT touch the live RL run / checkpoints. Run under the WSL hexgt-build venv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/mnt/e/Hexo-BotTrainer-hexgt")
for p in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    sys.path.insert(0, str(ROOT / "packages" / p / "python"))

import tempfile

import torch

import hexo_engine.api as eng
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_models.dense_cnn.compact_io import read_compact_shard
from hexo_models.hexgt.architecture import HexgtNetwork, VALUE_READOUT_MULT
from hexo_models.hexgt.config import (
    HexgtArchitectureConfig, HexgtConfig, HexgtSampleConfig,
    HexgtSelfPlayConfig, HexgtTrainingConfig,
)
from hexo_models.hexgt.inference import HexgtInference
from hexo_models.hexgt.mcts import new_mcts_session
from hexo_models.hexgt.selfplay import run_selfplay_games
from hexo_models.hexgt.trainer import HexgtTrainer


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {dev}  (torch {torch.__version__})")

cfg = HexgtConfig(
    architecture=HexgtArchitectureConfig(token_dim=64, gnn_layers=2, ctx_layers=2, ffn_dim=96, candidate_radius=3),
    selfplay=HexgtSelfPlayConfig(
        search_visits=48, widening_max_children=32, widening_min_children=2, max_actions=80,
        temperature=1.0, final_temperature=0.3, temperature_decay_moves=20, forced_playout_k=2.0,
        active_games=8,
    ),
    training=HexgtTrainingConfig(batch_size=16, warmup_steps=2, learning_rate=2e-4),
    samples=HexgtSampleConfig(soft_z_lambda=0.5),
)
a = cfg.architecture
model = HexgtNetwork(
    token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
    attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    short_term_value_horizons=(4, 12, 24),
).to(dev).eval()
print(f"value head in_features = {model.value_head[0].weight.shape[1]} "
      f"(= {VALUE_READOUT_MULT} x token_dim {a.token_dim}) -> global-pooled readout active")

# compile the search forward (the throughput-critical path) to confirm the pooled
# value + v3 features + injection/override compose under torch.compile on GPU.
if dev == "cuda":
    model.forward_policy_value = torch.compile(model.forward_policy_value, dynamic=True)

banner("1. SELF-PLAY (search runs Phase-4 injection + Phase-5 override) + soft-Z finalize")
tmp = Path(tempfile.mkdtemp(prefix="tss_smoke_"))
result = run_selfplay_games(
    model, cfg, num_games=6, output_dir=tmp, epoch=0, device=dev, fp16=(dev == "cuda"),
    base_seed=7, active_games=8, virtual_batch_size=16, collect_examples=2,
)
print(f"games done={result.completed_games + result.truncated_games} "
      f"searched_positions={result.searched_positions} raw_samples={result.raw_samples} "
      f"pos/s={result.positions_per_second:.1f}")
assert result.raw_samples > 0 and len(result.shard_paths) == 6

banner("2. SOFT-Z TARGETS in the shards (Phase 1): values must span (-1,1), not just +-1/0")
vals = []
for sp in result.shard_paths:
    for row in read_compact_shard(Path(sp)):
        vals.append(float(row.value))
import statistics
saturated = sum(1 for v in vals if abs(abs(v) - 1.0) < 1e-6 or abs(v) < 1e-6)
soft = [v for v in vals if 1e-6 < abs(v) < 1.0 - 1e-6]
print(f"value targets: n={len(vals)} mean={statistics.mean(vals):+.3f} "
      f"min={min(vals):+.3f} max={max(vals):+.3f} | "
      f"saturated(+-1/0)={saturated} soft(strictly interior)={len(soft)}")
assert soft, "soft-Z produced no interior value targets -- blend not applied"
print(f"-> soft-Z is recalibrating the label ({len(soft)}/{len(vals)} targets are strictly interior)")

banner("3. TRAINING STEP (Phase-2 pooled value head + Phase-3 v3 features in the loss)")
opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)
hist = trainer.train_on_shards([Path(p) for p in result.shard_paths], batch_size=16, max_steps=3)
assert hist, "no training steps"
h0 = hist[0]
print(f"train step keys: {sorted(h0)}")
print(f"step0: total={h0['total']:.4f} (finite={h0['total'] == h0['total']})")
assert h0["total"] == h0["total"], "loss is NaN"
print("-> end-to-end self-play -> soft-Z finalize -> train step runs on GPU, loss finite")

banner("4. TSS OVERRIDE/INJECTION fire in real GPU search")
inf = HexgtInference(model, device=dev, fp16=(dev == "cuda"))


def play(coords):
    s = eng.new_game()
    for q, r in coords:
        eng.apply_action(s, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return s


# forced-loss (two independent open fours, side to move lost) -> override drives -1
P0 = {0: (0, 0), 3: (1, 0), 4: (2, 0), 7: (3, 0), 8: (0, 3), 11: (1, 3), 12: (2, 3), 15: (3, 3), 16: (0, -2)}
P1 = {1: (0, 7), 2: (2, 7), 5: (4, 7), 6: (6, 7), 9: (0, 8), 10: (2, 8), 13: (4, 8), 14: (6, 8)}
flat = [None] * 17
for i, c in {**P0, **P1}.items():
    flat[i] = c
lost = play(flat)
kw = dict(visits=64, c_puct=1.5, seed=1, virtual_batch_size=8, widening_policy_mass=0.95,
          widening_max_children=16, widening_min_children=2, fpu_reduction=0.2,
          virtual_loss=1.0, root_policy_temperature=1.0)
sess = new_mcts_session(max_states=1 << 16, n=3)
rv = sess.run([0], [lost], inf, **kw)[0].root_value
print(f"forced-loss root_value = {rv:+.3f} (override should drive strongly negative)")
assert rv < -0.85, "override did not fire in GPU search"

banner("5. vA + vB OPTIMISM SUM harness (Phase 1+2 target; full trend needs a real run)")
# Same board, value from each side to move. With soft-Z + global pooling trained,
# this sum should trend toward 0. On a FRESH (barely-trained) model it just shows
# the harness works; the convergence requires a full RL run (flagged).
import hexo_models.hexgt.graph_build as gb


def value_of(state):
    batch = gb.batch_from_states([state], n=3)
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        logits = model.forward(batch)["value"]
    bins = torch.linspace(-1.0, 1.0, logits.shape[-1], device=logits.device)
    return float((torch.softmax(logits, dim=-1)[0] * bins).sum())


sums = []
for seed in (11, 23, 37, 51):
    s = eng.new_game(seed=seed)
    for _ in range(12):
        if eng.terminal(s) is not None:
            break
        acts = list(eng.legal_actions(s))
        eng.apply_action(s, acts[len(acts) // 2])
    if eng.terminal(s) is not None:
        continue
    # side-to-move value; then a near-identical board with the other side to move
    # (advance one neutral ply) for the opposing perspective.
    va = value_of(s)
    acts = list(eng.legal_actions(s))
    eng.apply_action(s, acts[len(acts) // 2])
    if eng.terminal(s) is not None:
        continue
    vb = value_of(s)
    sums.append(va + vb)
    print(f"  seed {seed}: v(A)={va:+.3f} v(next)={vb:+.3f} sum={va + vb:+.3f}")
if sums:
    print(f"mean |v(A)+v(B)| (fresh model, harness only) = {statistics.mean(abs(x) for x in sums):.3f}")
print("-> harness works; the documented +0.82 -> ~0 reduction requires a full RL run (flagged).")

banner("SMOKE COMPLETE — pipeline runs end-to-end on GPU with all 5 changes composed")
