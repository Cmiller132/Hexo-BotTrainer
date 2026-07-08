"""Strix-as-opponent-and-ANCHOR tests for the hexfield multi-stage eval (CPU-only).

Exercises the Strix zero-point wiring added to
``packages/hexfield/python/hexfield/multistage_eval.py``: the roster field, the
budget arm, the full-weight edge builder, and — the load-bearing behaviour —
that Strix is the PREFERRED pinned BT anchor (tier 0, above SealBot). The GPU
arena (``eval_arena.play_strix_match``, which needs torch + CUDA + the ``hexo_rs``
wheel) is mocked throughout via the ``play_strix_match`` injection seam, so this
collects and runs on a CPU-only interpreter.

Mirrors the conventions of ``test_hexfield_eval_orchestrator.py`` (the shared
``_make_run`` / ``_paired_match`` / ``_StubArena`` from ``hexfield_eval_kit``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(_REPO / "packages" / "hexfield" / "python"))

from hexfield import multistage_eval as mse  # noqa: E402
from hexfield.config import MultiStageEvalSection, parse_hexfield_config  # noqa: E402

from hexfield_eval_kit import (  # noqa: E402
    _StubArena,
    _make_run,
    _paired_match,
    _paired_match_from_scores,
)


# --------------------------------------------------------------------------- #
# Local helpers: a Strix arena + strix-enabled config.
# --------------------------------------------------------------------------- #
def _strix_match(n: int, winrate: float) -> dict:
    """A fake ``play_strix_match`` result — PAIRED CRN pentanomial, net-A-centric.

    Net A = hexfield candidate, net B = Strix. ``winrate`` is turned into a list
    of per-pair net-A scores (in {0, 1, 2}) that sum to ~the target win count,
    with a 1-split when the target is odd so the pair-level SE is non-zero. The
    shape (score + pentanomial.pairs + histogram) matches
    ``eval_arena._build_match_result`` + ``_pentanomial_block``, which
    ``_strix_edge`` / ``_build_strix_edge_from_match`` consume via
    ``_pentanomial_to_paired_result``.
    """

    n_pairs = max(1, n // 2)
    remaining = int(round(winrate * 2 * n_pairs))
    scores: list[int] = []
    for _ in range(n_pairs):
        s = 2 if remaining >= 2 else (1 if remaining == 1 else 0)
        scores.append(s)
        remaining -= s
    m = _paired_match_from_scores("cand", "strix", scores)
    m["meta"].update(
        {"games_requested": 2 * n_pairs, "visits": 512, "virtual_batch_size": 16,
         "kind": "hexfield_vs_strix", "paired_openings": True}
    )
    return m


class _StrixArena(_StubArena):
    """``_StubArena`` plus a ``play_strix_match`` seam (records calls).

    ``strix_winrate`` sets the candidate's binomial win rate vs Strix;
    ``strix_raises`` (callable or None) is invoked at the start of
    ``play_strix_match`` when set so a test can fail-open the Strix runner.
    """

    def __init__(self, *, strix_winrate: float = 0.6, strix_raises=None, **kw) -> None:
        super().__init__(**kw)
        self._strix_winrate = strix_winrate
        self._strix_raises = strix_raises
        self.strix_calls: list[tuple[str, int]] = []

    def play_strix_match(self, hexfield_ckpt, strix_ckpt, n, **kw) -> dict:
        if self._strix_raises is not None:
            self._strix_raises()
        # Positional (hexfield_ckpt, strix_ckpt, n) then keyword search knobs.
        self.strix_calls.append((kw.get("strix_label", "strix"), n))
        return _strix_match(n, self._strix_winrate)

    # The concurrent path plays checkpoint opponents through
    # play_multi_checkpoint_match; provide a fake that returns one paired match
    # per opponent (net-A = candidate).
    def play_multi_checkpoint_match(self, cand, opps, per, **kw) -> dict:
        out: dict[str, dict] = {}
        for label, _ckpt in opps:
            self.ckpt_calls.append((label, per))
            out[label] = _paired_match(kw["candidate_label"], label, max(1, per // 2), self._per_score)
        return out


def _strix_config(**overrides):
    """A no-SPRT config with Strix enabled (SealBot stays enabled by default)."""

    opp = {"strix_enabled": True, "strix_ckpt": "strix_ckpt.pt"}
    opp.update(overrides.pop("opponents", {}))
    return parse_hexfield_config(
        {"multi_stage_eval": {"sprt": {"enabled": False}, "opponents": opp, **overrides}}
    )


def _candidate(run: Path, epoch: int) -> Path:
    return run / "checkpoints" / f"epoch_{epoch:06d}.pt"


# =========================================================================== #
# 1. Roster + budget: Strix is a separate zero-point field, funded like SealBot.
# =========================================================================== #
def test_select_opponents_builds_strix_zero_point(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    roster = mse.select_opponents(
        run, _candidate(run, 40), _strix_config().multi_stage_eval, candidate_epoch=40
    )
    assert roster.strix is not None
    assert roster.strix.role == "strix"
    assert roster.strix.label == mse.STRIX_LABEL
    # Strix is a separate roster field, NOT one of the checkpoint opponents.
    assert all(o.role != "strix" for o in roster.opponents)
    assert roster.strix.label in roster.all_labels()


def test_strix_disabled_by_default(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    roster = mse.select_opponents(
        run, _candidate(run, 40), MultiStageEvalSection(), candidate_epoch=40
    )
    assert roster.strix is None
    assert mse.STRIX_LABEL not in roster.all_labels()


def test_allocate_budget_carves_strix_arm_only_when_enabled() -> None:
    # SealBot-only callers keep their exact dict shape (no STRIX key).
    base = mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)
    assert mse.STRIX_LABEL not in base
    assert base == {mse.SEALBOT_LABEL: 32, "per_checkpoint": 24}

    # With Strix on, both zero-points are carved before the checkpoint split.
    both = mse.allocate_budget(
        128, n_checkpoint_opponents=4, has_sealbot=True, has_strix=True, strix_share=0.25
    )
    assert both[mse.STRIX_LABEL] == 32
    assert both[mse.SEALBOT_LABEL] == 32
    assert both["per_checkpoint"] == 16  # (128 - 32 - 32) // 4

    # Strix floored to one pairing at a small positive budget.
    small = mse.allocate_budget(
        4, n_checkpoint_opponents=3, has_sealbot=True, has_strix=True, strix_share=0.25
    )
    assert small[mse.STRIX_LABEL] >= 2

    # Zero budget stays all-zeros (STRIX key present since enabled).
    zero = mse.allocate_budget(
        0, n_checkpoint_opponents=4, has_sealbot=True, has_strix=True, strix_share=0.25
    )
    assert zero == {mse.SEALBOT_LABEL: 0, "per_checkpoint": 0, mse.STRIX_LABEL: 0}


# =========================================================================== #
# 2. _choose_anchor: Strix is the PREFERRED tier-0 anchor, above SealBot.
# =========================================================================== #
def test_choose_anchor_prefers_strix_over_sealbot() -> None:
    cand = "cand_ep40"
    roster = mse.Roster(
        candidate_label=cand, candidate_epoch=40,
        sealbot=mse.Opponent(mse.SEALBOT_LABEL, "sealbot", None, None),
        strix=mse.Opponent(mse.STRIX_LABEL, "strix", Path("s"), None),
        champion=mse.Opponent("ep20", "champion", Path("e"), 20),
        opponents=(
            mse.Opponent("bc_prefit", "anchor", Path("b"), 2),
            mse.Opponent("ep20", "champion", Path("e"), 20),
        ),
    )
    BT = mse.eval_stats.BTEdge

    # Strix + SealBot both have edges -> Strix wins (tier 0).
    assert mse._choose_anchor(
        [BT(cand, mse.STRIX_LABEL, 10, 10, 1.0), BT(cand, mse.SEALBOT_LABEL, 10, 10, 0.5)], roster
    ) == mse.STRIX_LABEL
    # Strix has no edge this pool -> falls back to SealBot (tier 1).
    assert mse._choose_anchor(
        [BT(cand, mse.SEALBOT_LABEL, 10, 10, 0.5), BT(cand, "bc_prefit", 10, 10, 1.0)], roster
    ) == mse.SEALBOT_LABEL
    # Neither zero-point has an edge -> bc_prefit (tier 2).
    assert mse._choose_anchor([BT(cand, "bc_prefit", 10, 10, 1.0)], roster) == "bc_prefit"


def test_choose_anchor_respects_configurable_strix_label() -> None:
    """Anchor-tier checks key on roster.strix.label, not the STRIX_LABEL default."""

    cand = "cand_ep40"
    roster = mse.Roster(
        candidate_label=cand, candidate_epoch=40,
        sealbot=mse.Opponent(mse.SEALBOT_LABEL, "sealbot", None, None),
        strix=mse.Opponent("strix_v2", "strix", Path("s"), None),
        champion=None, opponents=(),
    )
    BT = mse.eval_stats.BTEdge
    assert mse._choose_anchor(
        [BT(cand, "strix_v2", 10, 10, 1.0), BT(cand, mse.SEALBOT_LABEL, 10, 10, 0.5)], roster
    ) == "strix_v2"


# =========================================================================== #
# 3. End-to-end: Strix is the pinned BT anchor at FULL weight (both run paths).
# =========================================================================== #
def _assert_strix_anchored(rep: dict) -> None:
    fit = rep["ratings"]["fit"]
    assert fit["converged"] is True
    assert fit["anchor"] == mse.STRIX_LABEL
    assert fit["anchor_is_strix"] is True
    assert fit["anchor_is_sealbot"] is False
    # The pinned anchor is flagged in the ratings table.
    strix_row = next(p for p in rep["ratings"]["players"] if p["label"] == mse.STRIX_LABEL)
    assert strix_row["is_anchor"] is True
    assert strix_row["elo"] == pytest.approx(0.0)  # pinned at 0 Elo
    # The Strix edge is descriptive, PAIRED (CRN pentanomial), FULL weight.
    strix_edge = next(e for e in rep["edges"] if e["kind"] == "strix")
    assert strix_edge["paired"] is True
    assert strix_edge["weight"] == pytest.approx(1.0)
    assert rep["meta"]["anchor"] == mse.STRIX_LABEL
    assert rep["strix_winrate_ci95"] is not None


def test_monolithic_run_pins_strix_as_anchor(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StrixArena(per_score=2, sealbot_winrate=0.6, strix_winrate=0.65)
    rep = mse.run_multistage_eval(
        run, _candidate(run, 40), _strix_config(), candidate_epoch=40,
        write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        play_strix_match=arena.play_strix_match,
    )
    assert arena.strix_calls, "Strix zero-point edge was never played"
    _assert_strix_anchored(rep)
    # SealBot is still played and pooled, just NOT the anchor.
    assert arena.sealbot_calls
    assert mse.SEALBOT_LABEL in {e["opponent"] for e in rep["edges"]}


def test_concurrent_run_pins_strix_as_anchor(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StrixArena(per_score=2, sealbot_winrate=0.6, strix_winrate=0.65)
    rep = mse.run_multistage_eval_concurrent(
        run, _candidate(run, 40), _strix_config(), candidate_epoch=40,
        write_diagnostics=False,
        play_multi_checkpoint_match=arena.play_multi_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        play_strix_match=arena.play_strix_match,
    )
    assert arena.strix_calls
    _assert_strix_anchored(rep)


def test_strix_edge_pooled_at_full_weight(tmp_path: Path) -> None:
    """The persisted pool row for Strix carries kind='strix' and weight 1.0."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StrixArena(per_score=2, sealbot_winrate=0.6, strix_winrate=0.65)
    mse.run_multistage_eval_concurrent(
        run, _candidate(run, 40), _strix_config(), candidate_epoch=40,
        write_diagnostics=True,
        play_multi_checkpoint_match=arena.play_multi_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        play_strix_match=arena.play_strix_match,
    )
    pool = mse._load_pool(run / "diagnostics" / "eval_pool.json")
    strix_rows = [r for r in pool["edges"] if r.get("kind") == "strix"]
    assert len(strix_rows) == 1
    assert strix_rows[0]["weight"] == pytest.approx(1.0)  # FULL weight
    # Contrast: the SealBot row is down-weighted (< 1.0).
    sb_rows = [r for r in pool["edges"] if r.get("kind") == "sealbot"]
    assert sb_rows and sb_rows[0]["weight"] < 1.0


# =========================================================================== #
# 4. Fail-open + degraded substitution when Strix is unavailable.
# =========================================================================== #
def test_strix_unavailable_falls_back_to_sealbot_and_degrades(tmp_path: Path) -> None:
    """When the Strix runner raises, its edge is dropped, the anchor re-pins to
    SealBot (a GRACEFUL re-basing), and Stage D is marked degraded."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom():
        raise ModuleNotFoundError("No module named 'hexo_rs'")

    arena = _StrixArena(per_score=2, sealbot_winrate=0.6, strix_raises=boom)
    rep = mse.run_multistage_eval_concurrent(
        run, _candidate(run, 40), _strix_config(), candidate_epoch=40,
        write_diagnostics=False,
        play_multi_checkpoint_match=arena.play_multi_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        play_strix_match=arena.play_strix_match,
    )
    # The exception did not propagate; the pool still fits and anchors on SealBot.
    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["ratings"]["fit"]["anchor"] == mse.SEALBOT_LABEL
    assert mse.STRIX_LABEL not in {e["opponent"] for e in rep["edges"]}
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert "hexo_rs" in stage_c["strix_unavailable"]
    stage_d = next(s for s in rep["stages"] if s["stage"] == "D_pool")
    assert stage_d["status"] == "degraded"
    assert stage_d["strix_substituted"] is True
    v = rep["verdict"]
    assert v["anchor_substituted"] is True
    assert v["substituted_from"] == mse.STRIX_LABEL
    assert v["substituted_to"] == mse.SEALBOT_LABEL
    assert v["degraded"] is True
    assert "GRACEFUL" in v["degraded_note"]


def test_strix_all_truncated_match_is_treated_as_unavailable(tmp_path: Path) -> None:
    """A 0-decided Strix match would pool a 0-0 edge that silently de-anchors the
    fit; _play_strix_opponent surfaces it as unavailable instead so the fallback
    fires cleanly."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _strix_config()

    def zero_decided(hexfield_ckpt, strix_ckpt, n, **kw):
        m = _strix_match(n, 0.5)
        m["score"] = {"completed": 0, "a_wins": 0, "b_wins": 0, "decided": 0}
        return m

    roster = mse.select_opponents(run, _candidate(run, 40), cfg.multi_stage_eval, candidate_epoch=40)
    edge, ci, unavail = mse._play_strix_opponent(
        cfg.multi_stage_eval, roster, _candidate(run, 40), cfg, 8,
        play_strix_match=zero_decided, diagnostics_dir=run / "diagnostics",
    )
    assert edge is None and ci is None
    assert unavail is not None and "no decided games" in unavail


# =========================================================================== #
# 5. Pure-eval invariant holds with Strix enabled.
# =========================================================================== #
def test_strix_run_is_pure_eval(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StrixArena(per_score=2, sealbot_winrate=0.6, strix_winrate=0.65)
    rep = mse.run_multistage_eval_concurrent(
        run, _candidate(run, 40), _strix_config(), candidate_epoch=40,
        write_diagnostics=False,
        play_multi_checkpoint_match=arena.play_multi_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        play_strix_match=arena.play_strix_match,
    )
    assert rep["meta"]["pure_eval"] is True
    assert rep["meta"]["gating_enabled"] is False
    assert rep["meta"]["promotion_enabled"] is False
    # The strix knobs are surfaced in the config summary for audit.
    opp_summary = rep["meta"]["config"]["opponents"]
    assert opp_summary["strix_enabled"] is True
    assert rep["meta"]["config"]["strix_weight"] == pytest.approx(1.0)
