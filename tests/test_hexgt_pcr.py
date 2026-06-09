"""KataGo Playout Cap Randomization (PCR) for hexgt self-play.

Covers the per-move full/fast coin, the per-subset search gating (full =
noise + forced playouts + temperature schedule; fast = clean + greedy), the
recording exclusion (only full-search positions become training rows ⇒ ~half
the rows), and that PCR-off self-play is byte-for-byte the pre-PCR single-call
path. Tiny CPU model + tiny search so it stays a fast gate. See
docs/analysis/HEXGT_PCR_KATAGO_MAPPING.md.
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


def _cfg(*, pcr_enabled, full_proportion=0.5, full_visits=6, fast_visits=2):
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
            search_visits=full_visits,
            widening_max_children=8,
            widening_min_children=2,
            max_actions=24,
            temperature=1.0,
            final_temperature=0.2,
            temperature_decay_moves=8,
            forced_playout_k=2.0,
            root_dirichlet_noise_enabled=True,
            root_dirichlet_total_alpha=6.6,
            root_dirichlet_noise_fraction=0.30,
            pcr_enabled=pcr_enabled,
            pcr_full_proportion=full_proportion,
            pcr_fast_visits=fast_visits,
        ),
        training=HexgtTrainingConfig(batch_size=8, warmup_steps=2, learning_rate=1e-3),
    )


def _model(cfg):
    from hexo_models.hexgt.architecture import HexgtNetwork

    a = cfg.architecture
    return HexgtNetwork(
        token_dim=a.token_dim, gnn_layers=a.gnn_layers, ctx_layers=a.ctx_layers,
        attention_heads=a.attention_heads, ffn_dim=a.ffn_dim,
    ).cpu().eval()


# --------------------------------------------------------------------------- #
# 1) The per-move coin                                                         #
# --------------------------------------------------------------------------- #
def test_pcr_coin_proportion_and_edges() -> None:
    from hexo_models.hexgt.selfplay import _pcr_is_full

    # p=1 -> always full; p=0 -> always fast (the disabled-style edges).
    assert all(_pcr_is_full(7, 0, g, m, 1.0) for g in range(20) for m in range(20))
    assert not any(_pcr_is_full(7, 0, g, m, 0.0) for g in range(20) for m in range(20))

    # p=0.5 -> empirical full-fraction near 0.5 over many (game, move) draws.
    draws = [_pcr_is_full(7, 3, g, m, 0.5) for g in range(80) for m in range(40)]
    frac = sum(draws) / len(draws)
    assert 0.45 < frac < 0.55, f"full fraction {frac} not near 0.5"

    # p=0.25 -> empirical near 0.25.
    draws25 = [_pcr_is_full(7, 3, g, m, 0.25) for g in range(80) for m in range(40)]
    frac25 = sum(draws25) / len(draws25)
    assert 0.21 < frac25 < 0.29, f"full fraction {frac25} not near 0.25"


def test_pcr_coin_deterministic_and_decorrelated() -> None:
    from hexo_models.hexgt.selfplay import _pcr_is_full

    # Deterministic in its inputs.
    assert _pcr_is_full(1, 2, 3, 4, 0.5) == _pcr_is_full(1, 2, 3, 4, 0.5)
    # Not a constant function of move index (varies across moves within a game).
    seq = [_pcr_is_full(11, 0, 5, m, 0.5) for m in range(24)]
    assert any(seq) and not all(seq), "coin must vary across moves"
    # Two different games at the same move are not perfectly correlated.
    a = [_pcr_is_full(11, 0, g, 7, 0.5) for g in range(60)]
    b = [_pcr_is_full(11, 0, g, 8, 0.5) for g in range(60)]
    assert a != b, "adjacent-move coins should differ across games"


# --------------------------------------------------------------------------- #
# 2) Per-subset search gating (full vs fast)                                   #
# --------------------------------------------------------------------------- #
def test_pcr_single_call_per_root_vectors_gate_full_vs_fast(monkeypatch) -> None:
    """PCR runs ONE batched call per round with per-root overrides: full roots get the
    full cap + Dirichlet noise (per_root_noise True) + forced playouts + a >0 temperature;
    fast roots get the fast cap + no noise + forced_k 0 + greedy (temp 0). The vectors
    are aligned per root (a full root has visits=6 AND noise AND forced_k=2.0 AND temp>0)."""
    _torch()
    from hexo_models.hexgt import mcts as mcts_mod
    from hexo_models.hexgt.selfplay import run_selfplay_games

    calls: list[dict] = []
    original_run = mcts_mod.HexgtMctsSession.run

    def capture(self, *args, **kwargs):
        calls.append({
            "per_visits": list(kwargs.get("per_root_visits") or []),
            "per_forced_k": list(kwargs.get("per_root_forced_playout_k") or []),
            "per_noise": list(kwargs.get("per_root_noise") or []),
            "temps": list(kwargs.get("move_temperatures") or []),
            "alpha": kwargs.get("root_dirichlet_total_alpha"),
            "fraction": kwargs.get("root_dirichlet_noise_fraction"),
        })
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(mcts_mod.HexgtMctsSession, "run", capture)
    cfg = _cfg(pcr_enabled=True, full_visits=6, fast_visits=2)
    run_selfplay_games(
        _model(cfg), cfg, num_games=6, output_dir=_tmp(monkeypatch),
        epoch=0, device="cpu", fp16=False, base_seed=5, active_games=6, virtual_batch_size=2,
    )

    # Every round is a single call carrying per-root vectors; the noise params are
    # passed at call level (gated per-root by per_noise).
    saw_full = saw_fast = False
    for c in calls:
        assert c["per_visits"], "PCR must pass per_root_visits (single batched call)"
        assert len(c["per_visits"]) == len(c["per_forced_k"]) == len(c["per_noise"]) == len(c["temps"])
        assert c["alpha"] == pytest.approx(6.6) and c["fraction"] == pytest.approx(0.30)
        assert set(c["per_visits"]) <= {6, 2}, "only the two configured caps may appear"
        for v, k, noise, t in zip(c["per_visits"], c["per_forced_k"], c["per_noise"], c["temps"]):
            if v == 6:  # FULL root
                saw_full = True
                assert noise is True, "full root must apply Dirichlet noise"
                assert k == pytest.approx(2.0), "full root keeps forced playouts"
                assert t > 0.0, "full root uses the temperature schedule"
            else:  # FAST root (v == 2)
                saw_fast = True
                assert noise is False, "fast root must NOT apply Dirichlet noise"
                assert k == 0.0, "fast root disables forced playouts"
                assert t == 0.0, "fast root plays greedily (temp 0)"
    assert saw_full and saw_fast, "expected both full and fast roots across the run"


def test_pcr_off_is_single_full_call_per_round(monkeypatch) -> None:
    """PCR off: one run() per round, all at search_visits, with noise + forced
    playouts (the pre-PCR path)."""
    _torch()
    from hexo_models.hexgt import mcts as mcts_mod
    from hexo_models.hexgt.selfplay import run_selfplay_games

    calls: list[dict] = []
    original_run = mcts_mod.HexgtMctsSession.run

    def capture(self, *args, **kwargs):
        calls.append({
            "visits": kwargs.get("visits"),
            "alpha": kwargs.get("root_dirichlet_total_alpha"),
            "forced_k": kwargs.get("forced_playout_k"),
        })
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(mcts_mod.HexgtMctsSession, "run", capture)
    cfg = _cfg(pcr_enabled=False, full_visits=6)
    run_selfplay_games(
        _model(cfg), cfg, num_games=3, output_dir=_tmp(monkeypatch), epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=3, virtual_batch_size=2,
    )
    assert calls, "expected search rounds"
    assert all(c["visits"] == 6 for c in calls), "PCR off: every search at full visits"
    assert all(c["alpha"] is not None for c in calls), "PCR off: noise on every call"
    assert all(c["forced_k"] == 2.0 for c in calls), "PCR off: forced playouts on every call"


# --------------------------------------------------------------------------- #
# 3) Recording exclusion: only full positions become rows (~half)             #
# --------------------------------------------------------------------------- #
def test_pcr_records_only_full_positions(tmp_path) -> None:
    _torch()
    from hexo_models.dense_cnn.compact_io import read_compact_shard
    from hexo_models.hexgt.selfplay import run_selfplay_games

    cfg = _cfg(pcr_enabled=True, full_proportion=0.5, full_visits=6, fast_visits=2)
    result = run_selfplay_games(
        _model(cfg), cfg, num_games=6, output_dir=tmp_path, epoch=0,
        device="cpu", fp16=False, base_seed=5, active_games=6, virtual_batch_size=2,
    )

    # Accounting identities.
    assert result.pcr_enabled is True
    assert result.full_search_count + result.fast_search_count == result.searched_positions
    assert result.recorded_positions == result.full_search_count
    assert result.fast_search_count > 0, "with p=0.5 some moves must be fast"

    # Recorded rows == sum of shard rows == raw_samples (no sanitization in a healthy run).
    total_rows = sum(len(read_compact_shard(Path(p))) for p in result.shard_paths)
    assert total_rows == result.raw_samples
    assert result.raw_samples == result.recorded_positions
    assert result.pcr_fast_rows_excluded == result.fast_search_count

    # Roughly half of searched positions are recorded (p=0.5).
    frac = result.recorded_positions / max(1, result.searched_positions)
    assert 0.30 < frac < 0.70, f"recorded fraction {frac} not near 0.5"
    # Avg visits/move sits between the two caps and below the full cap.
    assert cfg.selfplay.pcr_fast_visits <= result.mean_search_visits <= cfg.selfplay.search_visits


def test_pcr_halves_rows_vs_disabled(tmp_path) -> None:
    """Same games/seed: PCR records markedly fewer rows than the all-full run,
    while still playing every move (searched positions ~ unchanged)."""
    _torch()
    from hexo_models.hexgt.selfplay import run_selfplay_games

    common = dict(num_games=6, device="cpu", fp16=False, base_seed=5,
                  active_games=6, virtual_batch_size=2)

    off = _cfg(pcr_enabled=False, full_visits=6)
    r_off = run_selfplay_games(_model(off), off, output_dir=tmp_path / "off", epoch=0, **common)

    on = _cfg(pcr_enabled=True, full_proportion=0.5, full_visits=6, fast_visits=2)
    r_on = run_selfplay_games(_model(on), on, output_dir=tmp_path / "on", epoch=0, **common)

    # PCR-off records every searched position.
    assert r_off.recorded_positions == r_off.searched_positions
    assert r_off.raw_samples == r_off.searched_positions
    # PCR-on records meaningfully fewer rows (recorded ~ half) ...
    assert r_on.raw_samples < r_off.raw_samples
    assert r_on.recorded_positions < r_on.searched_positions
    # ... yet still plays a comparable number of total positions (games still
    # run to terminal); not a hard equality (full/fast caps perturb trajectories).
    assert r_on.searched_positions > 0.5 * r_off.searched_positions


def test_pcr_off_shards_unchanged_field_present(tmp_path) -> None:
    """A PCR-off result still carries the PCR fields (disabled state), and the
    shard row count equals searched positions (no exclusion)."""
    _torch()
    from hexo_models.hexgt.selfplay import run_selfplay_games

    cfg = _cfg(pcr_enabled=False, full_visits=6)
    r = run_selfplay_games(
        _model(cfg), cfg, num_games=3, output_dir=tmp_path, epoch=1,
        device="cpu", fp16=False, base_seed=9, active_games=3, virtual_batch_size=2,
    )
    d = r.as_dict()
    assert d["pcr_enabled"] is False
    assert d["fast_search_count"] == 0
    assert d["recorded_positions"] == d["searched_positions"]
    assert d["pcr_fast_rows_excluded"] == 0


# --------------------------------------------------------------------------- #
# 4) opp-policy target masking on fast next-ply (owner option (b))             #
# --------------------------------------------------------------------------- #
def test_opp_policy_masked_when_next_ply_is_fast() -> None:
    """`_future_opponent_policy(mask_from_fast=True)` returns an empty (masked)
    target when the next opponent move was a FAST search (pcr_full=False), and the
    real distribution when it was full. Masking off reproduces dense_cnn behavior."""
    from types import SimpleNamespace

    from hexo_models.dense_cnn.samples import _future_opponent_policy

    def d(player, policy, pcr_full):
        return (player, SimpleNamespace(policy=policy, metadata={"pcr_full": pcr_full}), 0.0)

    decisions = [
        d("player0", ((1, 1.0),), True),   # 0: next opp = idx1 (P1 full)
        d("player1", ((2, 1.0),), True),   # 1
        d("player0", ((3, 1.0),), True),   # 2: next opp = idx3 (P1 FAST) -> masked
        d("player1", ((4, 1.0),), False),  # 3 fast
        d("player0", ((5, 1.0),), True),   # 4: no future opponent
    ]
    # mask on: full-next -> present; fast-next -> masked; no-future -> none.
    assert _future_opponent_policy(decisions, 0, "player0", mask_from_fast=True) == (
        ((2, 1.0),), "future_opponent_mcts")
    assert _future_opponent_policy(decisions, 2, "player0", mask_from_fast=True) == (
        (), "fast_unrecorded_masked")
    assert _future_opponent_policy(decisions, 4, "player0", mask_from_fast=True) == ((), "none")
    # mask off (dense_cnn default): the fast next-ply is used as-is.
    assert _future_opponent_policy(decisions, 2, "player0", mask_from_fast=False) == (
        ((4, 1.0),), "future_opponent_mcts")


def test_selfplay_passes_opp_mask_only_when_pcr_enabled(monkeypatch) -> None:
    """Self-play passes mask_opp_from_fast=True to finalize iff PCR is enabled, so
    the non-PCR finalize path is byte-identical."""
    _torch()
    from hexo_models.hexgt import selfplay as sp_mod
    from hexo_models.hexgt.selfplay import run_selfplay_games

    seen: list[bool] = []
    original = sp_mod.finalize_game_samples

    def capture(*args, **kwargs):
        seen.append(bool(kwargs.get("mask_opp_from_fast", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(sp_mod, "finalize_game_samples", capture)

    cfg_on = _cfg(pcr_enabled=True, full_visits=6, fast_visits=2)
    run_selfplay_games(_model(cfg_on), cfg_on, num_games=2, output_dir=_tmp(monkeypatch),
                       epoch=0, device="cpu", fp16=False, base_seed=5, active_games=2, virtual_batch_size=2)
    assert seen and all(seen), "PCR on: every finalize call must mask fast opp targets"

    seen.clear()
    cfg_off = _cfg(pcr_enabled=False, full_visits=6)
    run_selfplay_games(_model(cfg_off), cfg_off, num_games=2, output_dir=_tmp(monkeypatch),
                       epoch=1, device="cpu", fp16=False, base_seed=5, active_games=2, virtual_batch_size=2)
    assert seen and not any(seen), "PCR off: finalize must NOT mask (dense_cnn-identical)"


def _tmp(monkeypatch):
    """A throwaway temp dir for tests that take monkeypatch but not tmp_path."""
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="hexgt_pcr_"))
    return d
