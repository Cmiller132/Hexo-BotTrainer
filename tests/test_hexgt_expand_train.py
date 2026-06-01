"""Phase-4 gates: expand-time targets consistent + a CPU training pass decreases loss.

Uses synthetic compact rows (duck-typed like dense_cnn's Model1SampleData) so the
test is portable. The real-shard target validation is run via
scripts/_validate_hexgt_expand.py against recorded dense_cnn shards.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch():
    return pytest.importorskip("torch")


def _eng():
    return importlib.import_module("hexo_engine.api")


def _synthetic_row(seed: int, steps: int, value: float = 0.5):
    """Play a legal random game and build a compact-row-like object for it."""

    eng = _eng()
    from hexo_engine.types import unpack_coord_id

    rng = random.Random(seed)
    state = eng.new_game(seed=seed)
    history = []
    idx = 0
    for _ in range(steps):
        if eng.terminal(state) is not None:
            break
        acts = list(eng.legal_actions(state))
        if not acts:
            break
        a = rng.choice(acts)
        c = unpack_coord_id(int(eng.action_id(a)))
        idx += 1
        history.append((int(c.q), int(c.r), None, None, idx, None, None))
        eng.apply_action(state, a)
    # visit policy over a subset of CURRENT legal moves (all in candidates at n=8)
    legal = list(eng.legal_action_ids(state))
    rng.shuffle(legal)
    chosen = legal[: max(1, len(legal) // 4)]
    policy = [(int(a), 1.0 + rng.random()) for a in chosen]
    opp = [(int(a), 1.0) for a in legal[:3]]
    return SimpleNamespace(
        placement_history=history,
        policy=policy,
        opp_policy=opp,
        value=value,
        short_term_value=((1, 0.3),),
    ), state


def test_expand_targets_consistent_no_dropped_mass_at_n8() -> None:
    _torch()
    from hexo_models.hexgt.expand import expand_row_to_graph

    row, _state = _synthetic_row(seed=4, steps=18)
    exp = expand_row_to_graph(row, n=8, horizons=(1, 4))

    total_visit = sum(w for _a, w in row.policy)
    # at n=8 candidate_set == legal set, so no visit mass is dropped
    assert exp.dropped_policy_mass < 1e-6, exp.dropped_policy_mass
    assert abs(float(exp.policy.sum()) - total_visit) < 1e-4
    # every visited action id is a candidate
    cand = set(int(x) for x in exp.graph.candidate_ids.tolist())
    assert all(int(a) in cand for a, _w in row.policy)
    # value + short-term-value targets reused from the row
    assert abs(exp.value - 0.5) < 1e-9
    assert abs(exp.stvalue[1] - 0.3) < 1e-9 and exp.stvalue_mask[1] == 1.0
    assert exp.stvalue[4] == 0.0 and exp.stvalue_mask[4] == 0.0


def test_build_training_batch_is_byte_stable() -> None:
    torch = _torch()
    from hexo_models.hexgt.expand import build_training_batch

    rows = [_synthetic_row(seed=s, steps=12)[0] for s in (1, 2, 3)]
    b1, t1 = build_training_batch(rows, n=8, horizons=(1,))
    b2, t2 = build_training_batch(rows, n=8, horizons=(1,))
    for key in ("node_feat", "edge_index", "candidate_index", "candidate_graph", "candidate_ids"):
        assert torch.equal(b1[key], b2[key]), f"{key} not byte-stable"
    assert torch.equal(t1["policy"], t2["policy"])


def test_cpu_training_pass_decreases_loss() -> None:
    torch = _torch()
    from hexo_models.hexgt.architecture import HexgtNetwork
    from hexo_models.hexgt.config import parse_hexgt_config
    from hexo_models.hexgt.trainer import HexgtTrainer

    rows = [_synthetic_row(seed=s, steps=random.Random(s).randint(8, 20))[0] for s in range(8)]
    cfg = parse_hexgt_config({
        "device": "cpu",
        "architecture": {"token_dim": 48, "gnn_layers": 2, "ctx_layers": 2, "ffn_dim": 64,
                         "short_term_value_horizons": [1], "candidate_radius": 8},
        "training": {"batch_size": 4, "learning_rate": 3e-3, "amp": False, "warmup_steps": 5},
    })
    model = HexgtNetwork(token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64,
                         short_term_value_horizons=(1,))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    trainer = HexgtTrainer(model=model, config=cfg, optimizer=opt)

    history = trainer.train_on_rows(rows, batch_size=4, epochs=20)
    assert history, "no training steps ran"
    first = history[0]["total"]
    last = min(h["total"] for h in history[-3:])
    assert last < first * 0.8, f"loss did not decrease: {first} -> {last}"
    assert trainer.sample_count > 0
    # train state round-trips
    state = trainer.train_state.to_dict()
    trainer.load_train_state(state)
    assert trainer.train_state.global_step == history.__len__() or trainer.train_state.global_step > 0
