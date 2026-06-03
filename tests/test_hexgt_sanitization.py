"""Auditable non-finite-logit sanitization (Task 2).

`HexgtInference._count_and_sanitize` is the operational guard that keeps a run
alive when the fp16 forward emits inf/NaN logits, but it must NOT do so silently:
it counts the events, and self-play FLAGS the affected search rounds so their
positions are excluded from the training shards (a sanitized near-uniform prior +
neutral value must not become a training target). These tests pin both halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_utils", "hexo_engine", "hexo_runner"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _torch():
    return pytest.importorskip("torch")


def _tiny_config():
    from hexo_models.hexgt.config import (
        HexgtArchitectureConfig,
        HexgtConfig,
        HexgtSelfPlayConfig,
        HexgtTrainingConfig,
    )

    return HexgtConfig(
        architecture=HexgtArchitectureConfig(
            token_dim=48, gnn_layers=2, ctx_layers=2, ffn_dim=64, candidate_radius=2
        ),
        selfplay=HexgtSelfPlayConfig(
            search_visits=6, widening_max_children=8, widening_min_children=2, max_actions=16
        ),
        training=HexgtTrainingConfig(batch_size=8, warmup_steps=2, learning_rate=1e-3),
    )


def _tiny_model(cfg):
    from hexo_models.hexgt.architecture import HexgtNetwork

    a = cfg.architecture
    return HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()


def test_count_and_sanitize_counts_replaces_and_attributes() -> None:
    """The guard replaces every non-finite logit with a finite neutral value, counts
    the events, and attributes them to the affected graphs (positions)."""

    torch = _torch()
    from hexo_models.hexgt.inference import HexgtInference

    cfg = _tiny_config()
    inf = HexgtInference(_tiny_model(cfg), device="cpu", fp16=False)
    assert inf.sanitized_logit_total == 0

    # 3 graphs; candidates [g0,g0,g1,g2]. Put a +inf and a NaN in graph 0's policy
    # and a -inf in graph 2's value row.
    cand_graph = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    policy = torch.tensor([float("inf"), float("nan"), 0.5, 0.2])
    value = torch.zeros((3, 4))
    value[2, 1] = float("-inf")

    out_p, out_v, n_bad = inf._count_and_sanitize(policy, value, cand_graph, num_graphs=3)
    assert n_bad == 3  # two bad policy logits + one bad value logit
    assert bool(torch.isfinite(out_p).all()) and bool(torch.isfinite(out_v).all())
    # Replacement convention: NaN->0, +inf->30, -inf->-30.
    assert out_p[0].item() == pytest.approx(30.0)
    assert out_p[1].item() == pytest.approx(0.0)
    assert out_v[2, 1].item() == pytest.approx(-30.0)
    # Counters: 3 logits, 1 forward call, graphs 0 and 2 affected (2 positions).
    assert inf.sanitized_logit_total == 3
    assert inf.sanitized_forward_calls == 1
    assert inf.sanitized_graph_total == 2

    # A clean batch leaves the tensors untouched and the counters unchanged.
    clean_p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    _, _, n_clean = inf._count_and_sanitize(clean_p, torch.zeros((3, 4)), cand_graph, 3)
    assert n_clean == 0
    assert inf.sanitized_logit_total == 3 and inf.sanitized_forward_calls == 1


def test_selfplay_excludes_sanitized_positions(tmp_path, monkeypatch) -> None:
    """When the evaluator sanitizes a logit during a search round, self-play flags
    that round and drops its positions from the written shards, surfacing the
    counts in the result (so the guard is auditable, not silent)."""

    torch = _torch()
    from hexo_models.dense_cnn.compact_io import compact_row_count
    from hexo_models.hexgt import inference as inference_mod
    from hexo_models.hexgt.selfplay import run_selfplay_games

    cfg = _tiny_config()
    model = _tiny_model(cfg)

    # Inject a +inf into the FIRST evaluator forward only, so exactly the first
    # search round is tainted (later rounds stay clean -> a mix of kept/dropped).
    real_forward = type(model).forward_policy_value
    state = {"calls": 0}

    def poisoned_forward(self, batch):
        out = real_forward(self, batch)
        state["calls"] += 1
        if state["calls"] == 1 and out["policy"].numel():
            out = {**out, "policy": out["policy"].clone()}
            out["policy"][0] = float("inf")
        return out

    monkeypatch.setattr(type(model), "forward_policy_value", poisoned_forward, raising=True)

    result = run_selfplay_games(
        model, cfg, num_games=2, output_dir=tmp_path, epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=2, virtual_batch_size=2,
    )

    # The guard fired and was recorded.
    assert result.sanitized_logit_events >= 1
    assert result.sanitized_positions >= 1
    assert result.sanitized_search_rounds >= 1
    assert result.sanitized_samples_excluded >= 1
    # as_dict surfaces the audit fields for diagnostics/dashboard.
    d = result.as_dict()
    for key in (
        "sanitized_logit_events", "sanitized_positions",
        "sanitized_search_rounds", "sanitized_samples_excluded",
    ):
        assert key in d

    # The excluded positions are genuinely absent from the shards: total written
    # rows == (finalized positions) - (excluded). raw_samples already counts only
    # written rows, so it must equal the sum of shard row counts.
    written = sum(compact_row_count(Path(p)) for p in result.shard_paths)
    assert written == result.raw_samples
